"""Aggregate per-clip EF predictions to per-video (dense all-clips average),
join the reference EF from FileList.csv, and produce a Bland-Altman plot.

Input CSV format (no header): filename, clip_index, prediction
  e.g.  0X100CF05D141FF143.avi,0,62.6652

Usage:
  python3 bland_altman_from_clips.py \
      output/uniformer_s_strong_aug_ema_seed0/test_predictions.csv \
      --data_dir /home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic \
      --out bland_altman_main
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "font.family": "serif",
    "axes.linewidth": 0.8, "savefig.bbox": "tight",
})

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--data_dir", required=True)
ap.add_argument("--out", default="bland_altman_main")
args = ap.parse_args()

# per-clip predictions (no header)
df = pd.read_csv(args.csv, header=None, names=["FN", "clip", "pred"])
def norm(fn): return fn if str(fn).endswith(".avi") else str(fn) + ".avi"
df["FN"] = df["FN"].apply(norm)

# average clips -> per-video prediction (dense all-clips)
per_video = df.groupby("FN")["pred"].mean().reset_index()
print(f"{len(per_video)} videos (from {len(df)} clip predictions)")

# join true EF
fl = pd.read_csv(f"{args.data_dir}/FileList.csv")
fl["FN"] = fl["FileName"].apply(norm)
per_video = per_video.merge(fl[["FN", "EF"]], on="FN", how="left")
per_video = per_video.dropna(subset=["EF"])
print(f"{len(per_video)} videos matched to reference EF")

true = per_video["EF"].values
pred = per_video["pred"].values

# ---- Bland-Altman ----
mean_ef = (pred + true) / 2.0
diff = pred - true
bias = diff.mean(); sd = diff.std(ddof=1)
loa_hi = bias + 1.96 * sd; loa_lo = bias - 1.96 * sd
within = np.mean((diff >= loa_lo) & (diff <= loa_hi)) * 100
A = np.vstack([mean_ef, np.ones_like(mean_ef)]).T
slope, intercept = np.linalg.lstsq(A, diff, rcond=None)[0]
# also report R2/MAE as a check this is the right model
r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)
mae = np.mean(np.abs(pred - true))

print(f"\ncheck: R2={r2:.3f}  MAE={mae:.2f}  (should match the model you dumped)")
print(f"bias (mean diff)        = {bias:+.2f} EF points")
print(f"SD of differences       = {sd:.2f}")
print(f"95% limits of agreement = [{loa_lo:+.2f}, {loa_hi:+.2f}]")
print(f"points within LoA       = {within:.1f}%")
print(f"proportional-bias slope = {slope:+.3f}")

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.scatter(mean_ef, diff, s=12, alpha=0.35, color="#4053d3",
           edgecolors="none", rasterized=True)
ax.axhspan(loa_lo, loa_hi, color="#e9ecef", alpha=0.4, zorder=0)
ax.axhline(bias, color="#c1121f", lw=1.6, label=f"bias = {bias:+.2f}")
ax.axhline(loa_hi, color="#6c757d", lw=1.3, ls="--",
           label=f"95% LoA = [{loa_lo:+.1f}, {loa_hi:+.1f}]")
ax.axhline(loa_lo, color="#6c757d", lw=1.3, ls="--")
ax.axhline(0, color="black", lw=0.6, alpha=0.4)
xs = np.linspace(mean_ef.min(), mean_ef.max(), 50)
ax.plot(xs, slope * xs + intercept, color="#2a9d8f", lw=1.4, ls=":",
        label=f"trend (slope {slope:+.2f})")
ax.set_xlabel("Mean of predicted and reference EF (%)")
ax.set_ylabel("Predicted $-$ reference EF (EF points)")
ax.set_title("Bland--Altman agreement")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(True, ls="--", alpha=0.3)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote {args.out}.pdf / .png")
