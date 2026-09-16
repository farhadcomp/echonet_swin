"""UniFormer-S EF regression with heteroscedastic beta-NLL uncertainty.

Instead of MSE on a single output, the head predicts TWO numbers per video:
  mu       = EF point estimate
  log_var  = log predictive variance (so sigma^2 = exp(log_var) > 0 always)

Trained with the beta-NLL loss (Seitzer et al., 2022, "On the Pitfalls of
Heteroscedastic Uncertainty Estimation"):

    nll_i = 0.5 * [ (y - mu)^2 * exp(-log_var) + log_var ]
    w_i   = exp(beta * log_var).detach()          # beta in [0,1], stop-grad
    loss  = mean( w_i * nll_i )

  beta = 0   -> plain Gaussian NLL (best sigma, but mu accuracy can suffer)
  beta = 0.5 -> recommended; recovers near-MSE mu accuracy AND useful sigma
  beta = 1   -> mu trains like MSE, sigma still learned

The point estimate (mu) is used for R2/MAE/RMSE exactly as before; the predicted
sigma gives per-sample uncertainty in ONE forward pass (no MC sampling), so it
can vary per patient by construction. We report coverage and corr(sigma,|error|)
directly from the predicted sigma.

This reuses the proven baseline recipe (GroupNorm, 4-ch patch embed, EMA,
discriminative LR, cosine schedule, augmentation, dense all-clips eval).

Smoke test (watch mu R2 climb like MSE, and sigma start tracking error):
  CUDA_VISIBLE_DEVICES=2 python3 -u video_uniformer_nll.py --data_dir ... \
    --output output/_smoke_nll --uniformer_weights ... --mask_source zero \
    --num_epochs 45 --warmup_epochs 8 --beta 0.5 --augment 2>&1 | grep -m4 "\[nll\]"
"""
import math
import os
import time
import copy

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import tqdm

import echonet


class ModelEMA:
    """Exponential moving average of model weights."""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        ema_sd = self.ema.state_dict()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_sd[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                ema_sd[k].copy_(v)


def replace_bn_with_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1
            gn = torch.nn.GroupNorm(num_groups=groups, num_channels=num_channels,
                                    eps=child.eps, affine=child.affine)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)


# ----- log_var safety clamp (prevents exp overflow / div explosion) -----
LOGVAR_MIN, LOGVAR_MAX = -6.0, 8.0   # sigma^2 in [~0.0025, ~2980]


def beta_nll_loss(mu, log_var, target, beta=0.5):
    log_var = log_var.clamp(LOGVAR_MIN, LOGVAR_MAX)
    inv_var = torch.exp(-log_var)
    nll = 0.5 * ((target - mu) ** 2 * inv_var + log_var)
    if beta > 0:
        # stop-gradient weight
        w = torch.exp(beta * log_var).detach()             
        nll = w * nll
    return nll.mean()


@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="EF")
@click.option("--model_name", type=str, default="uniformer_s")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--uniformer_weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=45)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--frames", type=int, default=36)
@click.option("--period", type=int, default=4)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=8)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--backbone_lr", type=float, default=1e-5)
@click.option("--head_lr", type=float, default=1e-4)
@click.option("--warmup_epochs", type=int, default=8)
@click.option("--grad_clip", type=float, default=1.0)
@click.option("--mask_source", type=str, default="zero")
@click.option("--mask_dir", type=str, default="MaskedVideos")
@click.option("--augment/--no_augment", default=True)
@click.option("--beta", type=float, default=0.5,
              help="Beta-NLL weighting (0=plain NLL, 0.5=recommended, 1=MSE-like mu).")
@click.option("--logvar_init", type=float, default=2.5,
              help="Init bias for log_var head (sigma^2~12, sigma~3.5 to match EF error scale).")
def run(data_dir=None, output=None, task="EF", model_name="uniformer_s",
        pretrained=True, weights=None, uniformer_weights=None, run_test=False,
        num_epochs=45, weight_decay=1e-4, frames=36, period=4,
        num_workers=4, batch_size=8, device=None, seed=0,
        backbone_lr=1e-5, head_lr=1e-4, warmup_epochs=8, grad_clip=1.0,
        mask_source="zero", mask_dir="MaskedVideos", augment=True,
        beta=0.5, logvar_init=2.5):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join("output", "uniformer_s_nll_seed{}".format(seed))
    os.makedirs(output, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # ---------------- Model: 2-output head (mu, log_var) ----------------
    print("Initializing UniFormer-S for beta-NLL EF regression", flush=True)
    import sys
    sys.path.insert(0, "/home/AD.UNLV.EDU/farhadik/tests/dynamic")
    from models.uniformer import uniformer_small

    model = uniformer_small(in_chans=4)

    if pretrained and uniformer_weights is not None:
        checkpoint = torch.load(uniformer_weights, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
        pe_key = "patch_embed1.proj.weight"
        if pe_key in state_dict:
            w = state_dict[pe_key]
            w_extra = w.mean(dim=1, keepdim=True)
            state_dict[pe_key] = torch.cat([w, w_extra], dim=1)
            print("  patch_embed1.proj: extended 3->4 channels.", flush=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Pretrained loaded. missing(head)={len(missing)} unexpected={len(unexpected)}", flush=True)

    # head outputs 2 values: [mu, log_var]
    head = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(512, 2))
    # mu bias ~ mean EF
    head[1].bias.data[0] = 55.6
    # log_var bias ~ sane sigma          
    head[1].bias.data[1] = logvar_init   
    # small init for the log_var weight row so early log_var is dominated by its bias
    head[1].weight.data[1, :] *= 0.01

    model.head = torch.nn.Identity()
    replace_bn_with_gn(model, num_groups=32)
    print("  BatchNorm -> GroupNorm done.", flush=True)
    model.head = head
    print("  2-output head (mu, log_var) attached.", flush=True)

    model = model.to(device)
    ema = ModelEMA(model, decay=0.999)
    print(f"  UniFormer-S on {device}; EMA decay=0.999; beta={beta}", flush=True)

    if weights is not None:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded weights from {weights}", flush=True)

    # ---------------- Optimizer (discriminative LR) ----------------
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if ("head" in name) else backbone_params).append(param)
    optim = torch.optim.AdamW(
        [{"params": backbone_params, "lr": backbone_lr},
         {"params": head_params, "lr": head_lr}], weight_decay=weight_decay)

    cosine_epochs = max(1, num_epochs - warmup_epochs)
    sched_warm = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cosine_epochs, eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [sched_warm, sched_cos], milestones=[warmup_epochs])

    # ---------------- Dataset ----------------
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=data_dir, split="train", add_mask=False))
    kwargs = {"target_type": task, "mean": mean, "std": std,
              "length": frames, "period": period,
              "add_mask": True, "mask_source": mask_source, "mask_dir": mask_dir}
    dataset = {}
    # NOTE: augment flag passed to train set (echo.py reads it); pad kept as in baseline
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12, augment=augment)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # ---------------- Training ----------------
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume, bestR2 = 0, -float("inf")
        try:
            ck = torch.load(os.path.join(output, "checkpoint.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["state_dict"]); optim.load_state_dict(ck["opt_dict"])
            scheduler.load_state_dict(ck["scheduler_dict"]); epoch_resume = ck["epoch"] + 1
            bestR2 = ck.get("best_r2", -float("inf"))
            f.write("Resuming from epoch {}\n".format(epoch_resume)); f.flush()
        except FileNotFoundError:
            f.write("Starting run from scratch\n"); f.flush()

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ["train", "val"]:
                start = time.time()
                ds = dataset[phase]
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers,
                    shuffle=(phase == "train"), pin_memory=(device.type == "cuda"),
                    drop_last=(phase == "train"))
                loss, yhat, y, _ = run_epoch(
                    model, loader, phase == "train", optim, device, beta=beta,
                    grad_clip=grad_clip,
                    ema_callback=(ema.update if (phase == "train") else None),
                    log_nll=(phase == "train"))
                r2 = sklearn.metrics.r2_score(y, yhat)
                f.write("{},{},{},{},{}\n".format(epoch, phase, loss, r2, time.time() - start)); f.flush()

            scheduler.step()

            # EMA val eval (mu-based R2)
            val_loader = torch.utils.data.DataLoader(
                dataset["val"], batch_size=batch_size, num_workers=num_workers,
                shuffle=False, pin_memory=(device.type == "cuda"))
            _, yhat_e, y_e, _ = run_epoch(ema.ema, val_loader, False, None, device, beta=beta)
            ema_r2 = sklearn.metrics.r2_score(y_e, yhat_e)
            raw_r2 = r2
            print(f"  raw val R2={raw_r2:.4f} | EMA val R2={ema_r2:.4f}", flush=True)
            if ema_r2 >= raw_r2:
                val_r2, best_state, tag = ema_r2, ema.ema.state_dict(), "EMA"
            else:
                val_r2, best_state, tag = raw_r2, model.state_dict(), "raw"

            save = {"epoch": epoch, "state_dict": best_state,
                    "raw_state_dict": model.state_dict(), "ema_state_dict": ema.ema.state_dict(),
                    "period": period, "frames": frames, "best_r2": bestR2,
                    "r2": val_r2, "opt_dict": optim.state_dict(),
                    "scheduler_dict": scheduler.state_dict()}
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if val_r2 > bestR2:
                torch.save(save, os.path.join(output, "best.pt"))
                bestR2 = val_r2
                print(f"  New best val R2={bestR2:.4f} ({tag}) epoch {epoch}", flush=True)

        if num_epochs != 0:
            ck = torch.load(os.path.join(output, "best.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["state_dict"])
            f.write("Best val R2 {:.4f} from epoch {}\n".format(ck["r2"], ck["epoch"])); f.flush()

        # ---------------- Test / val eval (mu + sigma) ----------------
        if run_test:
            for split in ["val", "test"]:
                # one-clip
                loader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"))
                _, yhat, y, sig = run_epoch(model, loader, False, None, device, beta=beta, return_sigma=True)
                _write_metrics(f, split, "one clip", y, yhat)
                _write_uncertainty(f, split, "one clip", y, yhat, sig)

                # dense all-clips
                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"))
                _, yhat_all, y, sig_all = run_epoch(
                    model, loader, False, None, device, beta=beta,
                    save_all=True, block_size=batch_size, return_sigma=True)
                yhat_mean = np.array([x.mean() for x in yhat_all])
                # combine per-clip sigma: average variance across clips, then sqrt
                sig_mean = np.array([np.sqrt((s ** 2).mean()) for s in sig_all])
                _write_metrics(f, split, "all clips", y, yhat_mean)
                _write_uncertainty(f, split, "all clips", y, yhat_mean, sig_mean)

                # save predictions + sigma for later recalibration / plots
                with open(os.path.join(output, "{}_pred_sigma.csv".format(split)), "w") as g:
                    g.write("true_ef,pred_mu,pred_sigma\n")
                    for a, b, c in zip(y, yhat_mean, sig_mean):
                        g.write("{:.4f},{:.4f},{:.4f}\n".format(a, b, c))


def _write_metrics(f, split, mode, y, yhat):
    r2 = echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)
    mae = echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)
    rmse = tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))
    f.write("{} ({}) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, mode, *r2))
    f.write("{} ({}) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, mode, *mae))
    f.write("{} ({}) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, mode, *rmse))
    f.flush()


def _write_uncertainty(f, split, mode, y, mu, sigma):
    err = np.abs(mu - y)
    corr = float(np.corrcoef(sigma, err)[0, 1]) if sigma.std() > 1e-9 else float("nan")
    cov95 = float(np.mean(err <= 1.959963985 * sigma))
    f.write("{} ({}) UNC: mean_sigma={:.2f} corr(sigma,|err|)={:+.3f} cov95={:.1f}%\n".format(
        split, mode, sigma.mean(), corr, cov95 * 100))
    f.flush()
    print(f"  [{split} {mode}] mean_sigma={sigma.mean():.2f} corr={corr:+.3f} cov95={cov95*100:.1f}%", flush=True)


def run_epoch(model, loader, train, optim, device, beta=0.5,
              save_all=False, block_size=None, grad_clip=1.0,
              ema_callback=None, log_nll=False, return_sigma=False):
    """One epoch. Head outputs (mu, log_var). Loss = beta-NLL. Point est = mu."""
    model.train(train)
    total, n = 0.0, 0
    yhat, y, sig_out = [], [], []
    printed = False

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(loader)) as pbar:
            for X, target in loader:
                y.append(target.numpy())
                X = X.to(device)
                target = target.to(device).float()

                average = (len(X.shape) == 6)
                if average:
                    b, ncl, c, fr, h, w = X.shape
                    X = X.view(-1, c, fr, h, w)

                X = torch.nn.functional.interpolate(
                    X, size=(X.shape[2], 224, 224), mode="trilinear", align_corners=False)

                # forward (optionally chunked for all-clips)
                if block_size is None:
                    out = model(X)                       # (B,2)
                else:
                    chunks = [model(X[j:j+block_size]) for j in range(0, X.shape[0], block_size)]
                    out = torch.cat(chunks, 0)
                mu = out[:, 0]
                log_var = out[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
                sigma = torch.exp(0.5 * log_var)

                if save_all:
                    yhat.append(mu.view(-1).cpu().detach().numpy())
                    if return_sigma:
                        sig_out.append(sigma.view(-1).cpu().detach().numpy())

                if average:
                    mu = mu.view(b, ncl, -1).mean(1).view(-1)
                    target_eff = target.view(-1)
                else:
                    target_eff = target.view(-1)

                if not save_all:
                    yhat.append(mu.view(-1).cpu().detach().numpy())
                    if return_sigma and not average:
                        sig_out.append(sigma.view(-1).cpu().detach().numpy())

                # loss only when not in pure all-clips averaging mode
                if not average:
                    loss = beta_nll_loss(mu.view(-1), log_var.view(-1), target_eff, beta=beta)
                else:
                    loss = torch.nn.functional.mse_loss(mu.view(-1), target_eff)  # eval proxy

                if train:
                    optim.zero_grad(); loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optim.step()
                    if ema_callback is not None:
                        ema_callback(model)

                if log_nll and not printed and not average:
                    with torch.no_grad():
                        e = (mu.view(-1) - target_eff).abs().cpu().numpy()
                        s = sigma.view(-1).cpu().numpy()
                        c = np.corrcoef(s, e)[0, 1] if s.std() > 1e-9 else float("nan")
                    print(f"    [nll] loss={loss.item():.3f} mean_mu={mu.mean():.1f} "
                          f"mean_sigma={s.mean():.2f} (range {s.min():.2f}-{s.max():.2f}) "
                          f"batch_corr(sigma,|err|)={c:+.3f}", flush=True)
                    printed = True

                total += loss.item() * X.size(0); n += X.size(0)
                pbar.set_postfix_str("{:.3f}".format(total / n)); pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
        sig_final = np.concatenate(sig_out) if (return_sigma and sig_out) else None
    else:
        sig_final = sig_out if return_sigma else None
    y = np.concatenate(y)
    return total / n, yhat, y, sig_final


if __name__ == "__main__":
    run()