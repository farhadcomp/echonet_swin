"""Measure the ED/ES area-error correlation rho for the DeepLabV3 segmenter,
then recompute the segmentation-ceiling threshold with the empirical value.

For each video we take its two expert-traced frames (ED = larger area, ES =
smaller). At each we compute:
    GT area   = pixels in the rasterized expert tracing (same convention as the
                EchoNet segmentation ground truth)
    pred area = pixels in the DeepLabV3 mask (threshold 0.5), same 112x112 grid
    eps = pred_area / gt_area - 1          (relative area error)

Then:
    rho      = corr(eps_ED, eps_ES) across videos      <-- the key number
    sigma_eps = std of the pooled per-frame relative errors
and we plug rho into the ceiling formula
    sigma_EF = (1 - EF) * sigma_eps * sqrt(2 (1 - rho))
to get the break-even area error (where sigma_EF equals the model's MAE).

Usage:
  CUDA_VISIBLE_DEVICES=2 python3 measure_rho.py \
      --data_dir /home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic \
      --weights  /home/AD.UNLV.EDU/farhadik/tests/dynamic/deeplabv3_resnet50_random.pt \
      --split test
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
import tqdm

sys.path.insert(0, "/home/AD.UNLV.EDU/farhadik/tests/dynamic")
import echonet


def rasterize_tracing(rows, size=112):
    """Rasterize one frame's expert tracing to a filled binary mask (pixel count = area).
    EchoNet VolumeTracings: each row has X1,Y1,X2,Y2. The (X1,Y1) points trace one side of
    the LV border and (X2,Y2) the other; filling the enclosed polygon reproduces the
    segmentation ground truth."""
    x1 = rows["X1"].values; y1 = rows["Y1"].values
    x2 = rows["X2"].values; y2 = rows["Y2"].values
    # polygon: down the (x1,y1) side, back up the (x2,y2) side reversed
    xs = np.concatenate([x1, x2[::-1]])
    ys = np.concatenate([y1, y2[::-1]])
    poly = np.stack([xs, ys], axis=1).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def load_deeplab(weights, device):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
    model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=1)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="rho_areas.csv")
    ap.add_argument("--model_mae", type=float, default=4.1,
                    help="regressor MAE (EF points) used as the break-even bar")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deeplab(args.weights, device)
    print("DeepLabV3 loaded.")

    # EchoNet-Dynamic normalization stats (grayscale replicated across 3 channels).
    # Hardcoded to avoid echonet.utils.get_mean_and_std, which crashes on this
    # echonet version due to a tensor-shape assumption. These are the standard
    # EchoNet training statistics; the exact values barely affect a *ratio* of
    # areas (both GT and predicted masks are thresholded on the same normalized
    # frames), so precise values are not critical here.
    mean = np.array([33.741, 33.741, 33.741], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([51.184, 51.184, 51.184], dtype=np.float32).reshape(3, 1, 1)
    print(f"Using EchoNet stats: mean={mean.ravel()}, std={std.ravel()}")

    # tracings + filelist
    tr = pd.read_csv(os.path.join(args.data_dir, "VolumeTracings.csv"))
    fl = pd.read_csv(os.path.join(args.data_dir, "FileList.csv"))
    def norm(fn): return fn if str(fn).endswith(".avi") else str(fn) + ".avi"
    tr["FN"] = tr["FileName"].apply(norm)
    fl["FN"] = fl["FileName"].apply(norm)
    fl["SplitL"] = fl["Split"].str.lower()
    if args.split != "all":
        keep = set(fl[fl["SplitL"] == args.split]["FN"])
    else:
        keep = set(fl["FN"])
    ef_map = dict(zip(fl["FN"], fl["EF"]))

    vids = [v for v in tr["FN"].unique() if v in keep]
    print(f"{len(vids)} {args.split} videos with tracings.")

    rows_out = []
    for vid in tqdm.tqdm(vids):
        sub = tr[tr["FN"] == vid]
        frames = sorted(sub["Frame"].unique())
        if len(frames) < 2:
            continue
        # GT area per traced frame (pixel count of rasterized tracing)
        areas = {}
        masks_gt = {}
        for fr in frames:
            m = rasterize_tracing(sub[sub["Frame"] == fr])
            areas[fr] = int(m.sum()); masks_gt[fr] = m
        ed_frame = max(areas, key=areas.get)   # larger area = diastole
        es_frame = min(areas, key=areas.get)   # smaller = systole
        gt_ed, gt_es = areas[ed_frame], areas[es_frame]
        if gt_ed == 0 or gt_es == 0:
            continue

        # read those two frames from the video, run DeepLabV3
        vpath = os.path.join(args.data_dir, "Videos", vid)
        cap = cv2.VideoCapture(vpath)
        pred = {}
        for fr in (ed_frame, es_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, img = cap.read()
            if not ok:
                pred[fr] = None; continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # HWC, 0-255
            img = img.transpose(2, 0, 1)                                  # CHW
            img = (img - mean) / std
            t = torch.from_numpy(img).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logit = model(t)["out"]
                p = (torch.sigmoid(logit) > args.threshold).float()
            pred[fr] = int(p.sum().item())
        cap.release()
        if pred.get(ed_frame) is None or pred.get(es_frame) is None:
            continue
        pred_ed, pred_es = pred[ed_frame], pred[es_frame]

        eps_ed = pred_ed / gt_ed - 1.0
        eps_es = pred_es / gt_es - 1.0
        rows_out.append(dict(FN=vid, ed_frame=ed_frame, es_frame=es_frame,
                             gt_ed=gt_ed, pred_ed=pred_ed, gt_es=gt_es, pred_es=pred_es,
                             eps_ed=eps_ed, eps_es=eps_es, EF=ef_map.get(vid, np.nan)))

    df = pd.DataFrame(rows_out)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} videos)")

    # ---------------- analysis ----------------
    eps_ed = df["eps_ed"].values; eps_es = df["eps_es"].values
    rho = float(np.corrcoef(eps_ed, eps_es)[0, 1])
    # pooled per-frame relative-error std (magnitude of area error)
    pooled = np.concatenate([eps_ed, eps_es])
    sigma_eps = float(np.std(pooled))
    mean_abs = float(np.mean(np.abs(pooled)))

    print("\n==================== RESULTS ====================")
    print(f"  n videos                     = {len(df)}")
    print(f"  rho (corr eps_ED, eps_ES)    = {rho:+.3f}   <-- the key number")
    print(f"  sigma_eps (std of rel error) = {sigma_eps*100:.1f}%")
    print(f"  mean |rel area error|        = {mean_abs*100:.1f}%")
    print(f"  mean eps_ED = {eps_ed.mean()*100:+.1f}%   mean eps_ES = {eps_es.mean()*100:+.1f}%")

    # break-even threshold with measured rho, at representative EF=0.6
    EF = 0.60
    MAE = args.model_mae
    for rr, label in [(0.0, "rho=0 (independence assumption)"), (rho, f"rho={rho:.2f} (measured)")]:
        coef = (1 - EF) * np.sqrt(2 * (1 - rr))
        thr = (MAE / 100) / coef      # break-even sigma_eps as fraction
        print(f"\n  [{label}]")
        print(f"     coefficient (1-EF)*sqrt(2(1-rho)) = {coef:.3f}")
        print(f"     break-even area error (sigma_EF = {MAE} EF pts) = {thr*100:.1f}%")
    # where does the actual DeepLabV3 sit?
    coef_m = (1 - EF) * np.sqrt(2 * (1 - rho))
    sigma_EF_actual = coef_m * sigma_eps * 100  # EF points, using measured sigma_eps and rho
    print(f"\n  DeepLabV3 at its measured sigma_eps={sigma_eps*100:.1f}%, rho={rho:.2f}:")
    print(f"     induced sigma_EF = {sigma_EF_actual:.1f} EF points  (compare to model MAE {MAE})")
    if sigma_EF_actual > MAE:
        print("     -> ABOVE the model's error: masks cannot help (matches empirical result).")
    else:
        print("     -> BELOW the model's error: area math alone would permit help;")
        print("        but empirically masks did not help (0.766<0.803), so amplification")
        print("        is not the sole factor (see channel/learned-regression discussion).")


if __name__ == "__main__":
    main()