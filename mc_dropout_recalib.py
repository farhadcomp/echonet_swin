"""MC-Dropout uncertainty WITH sigma-recalibration.

Builds directly on the proven mc_dropout_v3.py loop (same model load, same
one_clip_pass with the 224x224 interpolation, same head-dropout-active MC mode).

What it adds:
  * Runs MC-Dropout on BOTH validation and test splits.
  * Fits a single scalar variance-scaling factor s on VALIDATION:
       - RMS method (standard sigma-scaling): s_rms = sqrt(mean(z^2)), z=(y-mu)/sigma
       - Quantile method (targets 95% coverage): s_q = quantile_95(|z|)/1.96
  * Applies s to TEST sigma and reports calibrated coverage at 50/80/90/95%.
  * Saves a reliability diagram (coverage curve) and a recalibrated scatter.

Recalibration only RESCALES sigma; it does NOT change the mean prediction or the
ranking corr(sigma,|error|). It fixes coverage (calibration), not accuracy.

Run:
  CUDA_VISIBLE_DEVICES=2 python3 mc_dropout_recalib.py
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics

sys.path.insert(0, "/home/AD.UNLV.EDU/farhadik/tests/dynamic")
import echonet
from models.uniformer import uniformer_small
import importlib.util
spec = importlib.util.spec_from_file_location(
    "vu", "/home/AD.UNLV.EDU/farhadik/tests/dynamic/echonet/utils/video_uniformer_area.py")
vu = importlib.util.module_from_spec(spec); spec.loader.exec_module(vu)

# ============================ CONFIG ============================
DATA_DIR   = "/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic"
WEIGHTS    = "output/uniformer_s_strong_aug_ema_seed0/best.pt"
FRAMES, PERIOD = 36, 4
MC_SAMPLES = 20            # T passes
N_VAL      = None          # None = all 1288 val videos; or set e.g. 400 for a quick check
N_TEST     = None          # None = all 1276 test videos; or set e.g. 400
SEED       = 0
OUT_REL    = "mc_dropout_reliability.png"
OUT_SCAT   = "mc_dropout_scatter_calibrated.png"
OUT_CSV    = "mc_dropout_recalib_results.csv"
Z95        = 1.959963985   # 1.96
# ===============================================================

np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- build EXACTLY as training (keys verified to match in v3) ----
model = uniformer_small(in_chans=4, drop_path_rate=0.0)
ef_head = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(512, 1))
ef_head[1].bias.data[0] = 55.6
model.head = torch.nn.Identity()
vu.replace_bn_with_gn(model, num_groups=32)
model.head = ef_head
ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
state = ck["state_dict"]
if any(k.startswith("module.") for k in state):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.to(device)
print(f"Loaded {WEIGHTS}")

mean, std = echonet.utils.get_mean_and_std(
    echonet.datasets.Echo(root=DATA_DIR, split="train", add_mask=False))

def make_loader(split, n_videos):
    full = echonet.datasets.Echo(root=DATA_DIR, split=split, target_type="EF",
                                 mean=mean, std=std, length=FRAMES, period=PERIOD,
                                 add_mask=True, mask_source="zero")
    if n_videos is not None and n_videos < len(full):
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(full), n_videos, replace=False))
        ds = torch.utils.data.Subset(full, idx.tolist())
        print(f"  {split}: {n_videos} of {len(full)} videos (seed {SEED})")
    else:
        ds = full
        print(f"  {split}: ALL {len(full)} videos")
    return torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4,
                                       shuffle=False, pin_memory=True)

def set_mc_mode():
    """eval everywhere except dropout (forced train). Set ONCE; never re-eval."""
    model.eval()
    n = 0
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train(); n += 1
    return n

def one_clip_pass(loader):
    """One forward over a split, one clip/video. Interpolate 112->224 like run_epoch."""
    ys, ps = [], []
    with torch.no_grad():
        for X, t in loader:
            X = X.to(device).float()
            X = torch.nn.functional.interpolate(
                X, size=(X.shape[2], 224, 224), mode="trilinear", align_corners=False)
            out = model(X)
            out = out[0] if isinstance(out, tuple) else out
            ps.append(out.view(-1).cpu().numpy())
            ys.append(t.numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)

def mc_run(split, loader):
    """Deterministic sanity, then T MC passes. Returns y, mean, std."""
    model.eval()
    y0, p0 = one_clip_pass(loader)
    print(f"  [{split}] deterministic one-clip R2 = {sklearn.metrics.r2_score(y0,p0):.3f}")
    set_mc_mode()
    preds, yref = [], None
    for t in range(MC_SAMPLES):
        yt, pt = one_clip_pass(loader)
        if yref is None: yref = yt
        preds.append(pt)
        print(f"    [{split}] pass {t+1:2d}/{MC_SAMPLES}  R2={sklearn.metrics.r2_score(yt,pt):+.3f}", flush=True)
    P = np.stack(preds, 0)
    return yref, P.mean(0), P.std(0)

def coverage(y, mu, sigma, z):
    return np.mean(np.abs(y - mu) <= z * sigma)

# ---- run val and test ----
print("Running MC-Dropout on validation...")
yv, muv, sv = mc_run("val", make_loader("val", N_VAL))
print("Running MC-Dropout on test...")
yt, mut, st = mc_run("test", make_loader("test", N_TEST))

# guard against any zero sigma (shouldn't happen with head dropout live)
sv = np.clip(sv, 1e-6, None); st = np.clip(st, 1e-6, None)

# ---- fit scale on VALIDATION ----
zv = (yv - muv) / sv                      # standardized residuals on val
s_rms = float(np.sqrt(np.mean(zv**2)))    # standard variance scaling
s_q   = float(np.quantile(np.abs(zv), 0.95) / Z95)  # forces 95% val coverage

print("\n================ CALIBRATION (fit on validation) ================")
print(f"  raw val   95% coverage = {coverage(yv,muv,sv,Z95)*100:5.1f}%")
print(f"  s_rms (sqrt mean z^2)        = {s_rms:.3f}")
print(f"  s_q   (95% quantile / 1.96)  = {s_q:.3f}")
print(f"  val 95% coverage after s_rms = {coverage(yv,muv,s_rms*sv,Z95)*100:5.1f}%")
print(f"  val 95% coverage after s_q   = {coverage(yv,muv,s_q*sv,Z95)*100:5.1f}%")

# ---- apply to TEST, report coverage at several nominal levels ----
levels = [(0.50, 0.6744898), (0.80, 1.2815516), (0.90, 1.6448536), (0.95, Z95)]
def report(tag, sigma):
    r2  = sklearn.metrics.r2_score(yt, mut)
    mae = sklearn.metrics.mean_absolute_error(yt, mut)
    err = np.abs(mut - yt)
    corr = np.corrcoef(sigma, err)[0,1] if sigma.std()>1e-6 else float("nan")
    print(f"\n  [{tag}] mean sigma = {sigma.mean():.2f} EF pts   "
          f"corr(sigma,|err|) = {corr:+.3f}   (R2={r2:.3f}, MAE={mae:.2f})")
    for lvl, z in levels:
        print(f"      nominal {int(lvl*100):2d}%  ->  empirical {coverage(yt,mut,sigma,z)*100:5.1f}%")

print("\n================ TEST coverage ================")
report("raw", st)
report("calibrated (s_rms)", s_rms*st)
report("calibrated (s_q)",   s_q*st)

# ---- reliability diagram (test, raw vs calibrated s_rms) ----
nominal = np.linspace(0.05, 0.99, 40)
from scipy.stats import norm
zs = norm.ppf(0.5 + nominal/2)            # two-sided z for each nominal level
emp_raw = [coverage(yt,mut,st, z) for z in zs]
emp_cal = [coverage(yt,mut,s_rms*st, z) for z in zs]

plt.figure(figsize=(6,6))
plt.plot([0,1],[0,1],"k--",alpha=0.6,label="Perfect calibration")
plt.plot(nominal, emp_raw, "o-", color="indianred", ms=4, label="Raw MC-dropout")
plt.plot(nominal, emp_cal, "s-", color="seagreen", ms=4, label=f"Recalibrated (s={s_rms:.2f})")
plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage (test)")
plt.title("Reliability diagram: MC-dropout uncertainty")
plt.legend(loc="upper left"); plt.grid(True, ls="--", alpha=0.4)
plt.gca().set_aspect("equal","box"); plt.xlim(0,1); plt.ylim(0,1)
plt.tight_layout(); plt.savefig(OUT_REL, dpi=200, bbox_inches="tight")
print(f"\nwrote {OUT_REL}")

# ---- recalibrated scatter (subset for clarity) ----
rng = np.random.default_rng(SEED)
sel = rng.choice(len(yt), min(50, len(yt)), replace=False)
ys_, ms_, ss_ = yt[sel], mut[sel], (s_rms*st)[sel]
plt.figure(figsize=(8,8))
plt.errorbar(ys_, ms_, yerr=Z95*ss_, fmt="o", capsize=4, ecolor="red",
             markerfacecolor="royalblue", markeredgecolor="darkblue",
             alpha=0.75, markersize=6, elinewidth=1.2,
             label=f"Recalibrated MC-dropout (mean ± 1.96σ), n={len(sel)}")
lo = min(ys_.min(), ms_.min())-5; hi = max(ys_.max(), ms_.max())+5
plt.plot([lo,hi],[lo,hi],"k--",alpha=0.5,label="Perfect prediction")
plt.xlim(lo,hi); plt.ylim(lo,hi); plt.gca().set_aspect("equal","box")
plt.xlabel("True EF (%)"); plt.ylabel("Predicted EF (%)")
plt.title(f"Recalibrated MC-dropout (scale s={s_rms:.2f})")
plt.grid(True, ls="--", alpha=0.4); plt.legend(loc="upper left")
plt.tight_layout(); plt.savefig(OUT_SCAT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT_SCAT}")

# ---- dump CSV ----
with open(OUT_CSV,"w") as g:
    g.write("split,true_ef,pred_mean,sigma_raw,sigma_cal_rms\n")
    for a,b,c in zip(yv,muv,sv):
        g.write(f"val,{a:.4f},{b:.4f},{c:.4f},{s_rms*c:.4f}\n")
    for a,b,c in zip(yt,mut,st):
        g.write(f"test,{a:.4f},{b:.4f},{c:.4f},{s_rms*c:.4f}\n")
print(f"wrote {OUT_CSV}")
print(f"\nPaper numbers: scale s_rms={s_rms:.2f} (fit on val); "
      f"test 95% coverage raw {coverage(yt,mut,st,Z95)*100:.0f}% -> "
      f"calibrated {coverage(yt,mut,s_rms*st,Z95)*100:.0f}%.")
