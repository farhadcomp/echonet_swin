"""UniFormer-S EF prediction with optional area-consistency auxiliary task.

Two modes, controlled by --use_area_loss:
  OFF (default): EF-only, identical to the working EMA baseline.
  ON: adds a per-bin area head + three-term loss
        L = MSE(EF) + lambda_area * MSE(area@ED/ES bins) + lambda_smooth * smoothness(area_seq)
      motivated by the segmentation-wall finding (per-frame area variance
      corrupts the EF volume-difference; smoothing the internal area estimate
      attacks that variance directly).
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
import torchvision
import tqdm

import echonet


class ModelEMA:
    """Exponential moving average of model weights (single averaged model)."""
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


class UniFormerWithArea(torch.nn.Module):
    """UniFormer -> (EF scalar, per-bin area sequence (B, T')).

    EF head: global-pool features (T,H,W) -> (B,512) -> EF.
    Area head: spatial-pool only -> (B,512,T') -> per-bin area (B,T').
    """
    def __init__(self, backbone, ef_head):
        super().__init__()
        self.backbone = backbone
        self.ef_head = ef_head
        self.area_head = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.backbone.forward_features(x)   # (B, 512, T', 7, 7)
        ef_vec = feat.flatten(2).mean(-1)          # (B, 512)
        ef = self.ef_head(ef_vec).view(-1)         # (B,)
        area_feat = feat.mean(dim=(3, 4))          # (B, 512, T')
        area_feat = area_feat.permute(0, 2, 1)     # (B, T', 512)
        area_seq = self.area_head(area_feat).squeeze(-1)  # (B, T')
        return ef, area_seq


def replace_bn_with_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1
            gn = torch.nn.GroupNorm(
                num_groups=groups, num_channels=num_channels,
                eps=child.eps, affine=child.affine,
            )
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)


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
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--frames", type=int, default=36)
@click.option("--period", type=int, default=4)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=16)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--backbone_lr", type=float, default=1e-5)
@click.option("--head_lr", type=float, default=5e-4)
@click.option("--warmup_epochs", type=int, default=5)
@click.option("--grad_clip", type=float, default=1.0)
@click.option("--mask_source", type=str, default="gt")
@click.option("--mask_dir", type=str, default="MaskedVideos")
@click.option("--use_area_loss/--no_area_loss", default=False,
              help="Enable the area-amplitude consistency auxiliary task (Option B).")
@click.option("--lambda_amp", type=float, default=0.3,
              help="Weight of the amplitude-consistency term (EF-from-area-curve vs EF).")
@click.option("--lambda_smooth", type=float, default=0.01,
              help="Light smoothness on the area sequence (suppresses single-bin spikes only).")
@click.option("--lambda_anchor", type=float, default=0.05,
              help="Weak anchor tying mean predicted area to mean GT area (keeps units sane).")
@click.option("--amp_beta", type=float, default=8.0,
              help="Sharpness of soft-max/soft-min in the amplitude estimate.")
@click.option("--amp_warmup_epochs", type=int, default=3,
              help="Epochs to train area head on anchor+smooth only before enabling amplitude loss.")
@click.option("--n_bins", type=int, default=18,
              help="Temporal bins T' produced by the backbone (36 frames -> 18).")
@click.option("--dense_eval/--strided_eval", default=False,
              help="Use dense every-start all-clips at test time (original EchoNet protocol, comparable to R(2+1)D).")
def run(
    data_dir=None, output=None, task="EF", model_name="uniformer_s",
    pretrained=True, weights=None, uniformer_weights=None, run_test=False,
    num_epochs=45, lr=1e-4, weight_decay=1e-4, frames=36, period=4,
    num_train_patients=None, num_workers=4, batch_size=16, device=None, seed=0,
    backbone_lr=1e-5, head_lr=5e-4, warmup_epochs=5, grad_clip=1.0,
    mask_source="gt", mask_dir="MaskedVideos",
    use_area_loss=False, lambda_amp=0.3, lambda_smooth=0.01,
    lambda_anchor=0.05, amp_beta=8.0, amp_warmup_epochs=3, n_bins=18,
    dense_eval=False,
):
    """Trains/tests EF prediction model."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join(
            "output", "video",
            "{}_{}_{}_{}".format(model_name, frames, period,
                                 "pretrained" if pretrained else "random"),
        )
    os.makedirs(output, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    if model_name == "uniformer_s":
        print("Initializing UniFormer-S for EF regression", flush=True)

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
                print("  patch_embed1.proj: extended 3->4 channels (mask channel init by RGB mean).", flush=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Pretrained weights loaded. missing(head)={len(missing)} unexpected={len(unexpected)}", flush=True)
        elif pretrained and uniformer_weights is None:
            print("WARNING: --pretrained set but --uniformer_weights not provided.", flush=True)

        # EF regression head
        ef_head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 1),
        )
        ef_head[1].bias.data[0] = 55.6

        # Convert BN -> GN on the BACKBONE first (model.head currently unused)
        model.head = torch.nn.Identity()
        replace_bn_with_gn(model, num_groups=32)
        print("  BatchNorm -> GroupNorm conversion done.", flush=True)

        if use_area_loss:
            # wrap: returns (ef, area_seq)
            model = UniFormerWithArea(model, ef_head)
            print("  Wrapped with per-bin area head (area-consistency ON).", flush=True)
        else:
            # EF-only: attach EF head as the model head, plain scalar output
            model.head = ef_head
            print("  EF-only mode (area-consistency OFF).", flush=True)

    elif model_name in ("swin3d_s", "swin3d_t"):
        print(f"Initializing {model_name.upper()} for EF regression", flush=True)
        pretrained_weights = "KINETICS400_V1" if pretrained else None
        if model_name == "swin3d_t":
            model = torchvision.models.video.swin3d_t(weights=pretrained_weights)
        else:
            model = torchvision.models.video.swin3d_s(weights=pretrained_weights)
        in_features = model.head.in_features
        model.head = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(in_features, 1))
        model.head[1].bias.data[0] = 55.6
    else:
        print(f"Initializing original video model: {model_name}", flush=True)
        model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.fc.bias.data[0] = 55.6

    if model_name == "uniformer_s":
        model = model.to(device)
        print(f"  UniFormer-S on single GPU: {device}", flush=True)
        ema = ModelEMA(model, decay=0.999)
        print("  EMA initialized (decay=0.999)", flush=True)
    else:
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        ema = None

    if weights is not None:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded weights from {weights}", flush=True)

    # ------------------------------------------------------------------ #
    # Optimizer (differential LR)
    # ------------------------------------------------------------------ #
    print(f"Optimizer: AdamW | backbone_lr={backbone_lr} | head_lr={head_lr} | wd={weight_decay}", flush=True)
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    optim = torch.optim.AdamW(
        [{"params": backbone_params, "lr": backbone_lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay,
    )

    cosine_epochs = max(1, num_epochs - warmup_epochs)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cosine_epochs, eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    print(f"Scheduler: LinearWarmup({warmup_epochs}) -> Cosine({cosine_epochs}, eta_min=1e-7)", flush=True)

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=data_dir, split="train", add_mask=False))
    kwargs = {
        "target_type": task, "mean": mean, "std": std,
        "length": frames, "period": period,
        "add_mask": True, "mask_source": mask_source, "mask_dir": mask_dir,
        "add_area_target": use_area_loss, "n_bins": n_bins,
    }
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12, augment=True)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)

    area_cfg = dict(use=use_area_loss, l_amp=lambda_amp, l_smooth=lambda_smooth,
                    l_anchor=lambda_anchor, beta=amp_beta, n_bins=n_bins,
                    amp_warmup=amp_warmup_epochs)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestR2 = -float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"),
                                    map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            optim.load_state_dict(checkpoint["opt_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            bestR2 = checkpoint.get("best_r2", -float("inf"))
            f.write("Resuming from epoch {}\n".format(epoch_resume)); f.flush()
            print(f"Resuming from epoch {epoch_resume}, best val R2={bestR2:.4f}", flush=True)
        except FileNotFoundError:
            f.write("Starting run from scratch\n"); f.flush()
            print("Starting run from scratch", flush=True)

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)

            for phase in ["train", "val"]:
                start_time = time.time()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers,
                    shuffle=(phase == "train"), pin_memory=(device.type == "cuda"),
                    drop_last=(phase == "train"))

                loss, yhat, y = run_epoch(
                    model, dataloader, phase == "train", optim, device,
                    model_name=model_name, grad_clip=grad_clip,
                    ema_callback=(ema.update if (phase == "train" and ema is not None) else None),
                    area_cfg=area_cfg,
                    log_area=(phase == "train" and area_cfg["use"]),
                    cur_epoch=epoch)

                if device.type == "cuda":
                    max_allocated = torch.cuda.max_memory_allocated(device)
                    max_reserved = torch.cuda.max_memory_reserved(device)
                else:
                    max_allocated = max_reserved = 0

                epoch_r2 = sklearn.metrics.r2_score(y, yhat)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(
                    epoch, phase, loss, epoch_r2, time.time() - start_time,
                    y.size, max_allocated, max_reserved, batch_size))
                f.flush()

            scheduler.step()
            current_lrs = [pg["lr"] for pg in optim.param_groups]
            print(f"  LRs: backbone={current_lrs[0]:.2e}, head={current_lrs[1]:.2e}", flush=True)

            # EMA val eval
            if ema is not None:
                val_loader_ema = torch.utils.data.DataLoader(
                    dataset["val"], batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"))
                _, yhat_ema, y_ema = run_epoch(
                    ema.ema, val_loader_ema, False, None, device,
                    model_name=model_name, grad_clip=0, area_cfg=area_cfg)
                ema_val_r2 = sklearn.metrics.r2_score(y_ema, yhat_ema)
                raw_val_r2 = epoch_r2
                print(f"  raw val R2={raw_val_r2:.4f} | EMA val R2={ema_val_r2:.4f}", flush=True)
                if ema_val_r2 >= raw_val_r2:
                    val_r2, best_state, tag = ema_val_r2, ema.ema.state_dict(), "EMA"
                else:
                    val_r2, best_state, tag = raw_val_r2, model.state_dict(), "raw"
            else:
                val_r2, best_state, tag = epoch_r2, model.state_dict(), "raw"

            save = {
                "epoch": epoch, "state_dict": best_state,
                "raw_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict() if ema is not None else None,
                "period": period, "frames": frames, "best_r2": bestR2,
                "loss": loss, "r2": val_r2,
                "opt_dict": optim.state_dict(), "scheduler_dict": scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if val_r2 > bestR2:
                torch.save(save, os.path.join(output, "best.pt"))
                bestR2 = val_r2
                print(f"  New best val R2={bestR2:.4f} ({tag}) at epoch {epoch} -- saved best.pt", flush=True)

        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"),
                                    map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            f.write("Best val R2 {:.4f} from epoch {}\n".format(checkpoint["r2"], checkpoint["epoch"]))
            f.flush()
            print(f"Loaded best.pt: val R2={checkpoint['r2']:.4f} from epoch {checkpoint['epoch']}", flush=True)

        # ------------------------------------------------------------------ #
        # Test / val evaluation
        # ------------------------------------------------------------------ #
        if run_test:
            for split in ["val", "test"]:
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = run_epoch(model, dataloader, False, None, device,
                                          model_name=model_name, grad_clip=0, area_cfg=area_cfg)
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *tuple(map(math.sqrt, echonet.utils.bootstrap(
                        y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all", dense_clips=dense_eval)
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = run_epoch(model, dataloader, False, None, device,
                                          save_all=True, block_size=batch_size,
                                          model_name=model_name, grad_clip=0, area_cfg=area_cfg)
                yhat_mean = np.array(list(map(lambda x: x.mean(), yhat)))

                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat_mean, sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat_mean, sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *tuple(map(math.sqrt, echonet.utils.bootstrap(
                        y, yhat_mean, sklearn.metrics.mean_squared_error)))))
                f.flush()

                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for filename, pred in zip(ds.fnames, yhat):
                        for i, p in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))

                echonet.utils.latexify()
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat_mean.min()); upper = max(y.max(), yhat_mean.max())
                plt.scatter(y, yhat_mean, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)"); plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80]); plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split))); plt.close(fig)

                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat_mean)
                    auc_score = sklearn.metrics.roc_auc_score(y > thresh, yhat_mean)
                    print(thresh, auc_score); plt.plot(fpr, tpr)
                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split))); plt.close(fig)


def _unpack_ef(out):
    """Model may return (ef, area_seq) or just ef. Always return (ef, area_seq_or_None)."""
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def soft_amplitude_ef(area_seq, beta=8.0, eps=1e-4):
    """Differentiable EF-from-area-curve via soft-max/soft-min over the sequence.

    EF is defined by the contraction amplitude: (V_max - V_min) / V_max.
    For a clip spanning several beats, the soft-max approximates V_ED (the
    largest area reached) and the soft-min approximates V_ES (the smallest).
    Their normalised difference is an EF estimate derived purely from the
    predicted area *shape* — independent of the direct EF head.

    Hard max/min pass gradient only to one bin; the softmax-weighted version
    lets gradient flow to all bins, so the whole curve is shaped.

    Args:
        area_seq: (B, T) predicted per-bin areas (model units, ~0.3..2.0).
        beta:     sharpness of the soft extrema (higher = closer to hard max/min).
    Returns:
        (B,) soft EF estimate in [0,1]-ish range (×100 done by caller if needed).
    """
    # soft max: sum(area * softmax(beta*area))
    w_max = torch.softmax(beta * area_seq, dim=1)
    soft_max = (area_seq * w_max).sum(dim=1)
    # soft min: sum(area * softmax(-beta*area))
    w_min = torch.softmax(-beta * area_seq, dim=1)
    soft_min = (area_seq * w_min).sum(dim=1)
    # Denominator clamped to a positive floor so an untrained head (tiny/mixed
    # area values at init) cannot blow up the ratio. 0.2 ~= 200px floor.
    denom = soft_max.clamp(min=0.2) + eps
    ef = (soft_max - soft_min) / denom
    # Clamp to a sane EF range so a transient bad curve can't explode the loss.
    return (ef * 100.0).clamp(-50.0, 150.0)


def run_epoch(model, dataloader, train, optim, device,
              save_all=False, block_size=None, model_name="uniformer_s",
              grad_clip=1.0, ema_callback=None, area_cfg=None, log_area=False,
              cur_epoch=999):
    """One epoch. Handles both EF-only and (ef, area_seq) models.

    When area_cfg['use'] is True, the dataloader yields (X, (ef_true, area_tgt))
    and the loss is EF + lambda_area*area_sup + lambda_smooth*smoothness.
    Otherwise it yields (X, ef_true) and the loss is plain EF MSE.
    """
    use_area = bool(area_cfg and area_cfg.get("use", False))
    l_amp = area_cfg.get("l_amp", 0.3) if area_cfg else 0.3
    l_smooth = area_cfg.get("l_smooth", 0.01) if area_cfg else 0.01
    l_anchor = area_cfg.get("l_anchor", 0.05) if area_cfg else 0.05
    amp_beta = area_cfg.get("beta", 8.0) if area_cfg else 8.0
    amp_warmup = area_cfg.get("amp_warmup", 3) if area_cfg else 3
    amp_active = cur_epoch >= amp_warmup    # amplitude loss only after warmup

    model.train(train)
    total = 0.0; n = 0; s1 = 0.0; s2 = 0.0
    yhat = []; y = []
    printed_area = False

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for X, target in dataloader:

                # ---- unpack target ----
                if use_area:
                    ef_true, area_tgt = target            # area_tgt: (B,5)
                else:
                    ef_true = target
                    area_tgt = None

                y.append(ef_true.numpy())
                X = X.to(device)
                ef_true = ef_true.to(device).float()
                if area_tgt is not None:
                    area_tgt = area_tgt.to(device).float()

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, fr, h, w = X.shape
                    X = X.view(-1, c, fr, h, w)
                elif len(X.shape) == 5:
                    pass
                else:
                    raise ValueError(f"Unexpected input shape: {X.shape}")

                if model_name in ("uniformer_s", "swin3d_s", "swin3d_t"):
                    X = torch.nn.functional.interpolate(
                        X, size=(X.shape[2], 224, 224), mode="trilinear", align_corners=False)

                # for EF stats we use the (possibly clip-averaged) EF true
                ef_flat = ef_true.view(-1)
                s1 += ef_flat.sum().item()
                s2 += (ef_flat ** 2).sum().item()

                # ---- forward ----
                area_seq_full = None
                if block_size is None:
                    out = model(X)
                    ef_pred, area_seq_full = _unpack_ef(out)
                else:
                    ef_chunks = []
                    for j in range(0, X.shape[0], block_size):
                        out = model(X[j:j + block_size, ...])
                        ef_c, _ = _unpack_ef(out)
                        ef_chunks.append(ef_c)
                    ef_pred = torch.cat(ef_chunks)

                # ---- save predictions ----
                if save_all:
                    yhat.append(ef_pred.view(-1).to("cpu").detach().numpy())

                if average:
                    ef_pred = ef_pred.view(batch, n_clips, -1).mean(1).view(-1)

                if not save_all:
                    yhat.append(ef_pred.view(-1).to("cpu").detach().numpy())

                # ---- loss ----
                loss_ef = torch.nn.functional.mse_loss(ef_pred.view(-1), ef_true.view(-1))
                loss = loss_ef

                if use_area and (area_seq_full is not None) and (not average):
                    # ---- Option B: amplitude-consistency ----
                    # EF-from-area-curve via differentiable soft-max/min amplitude.
                    # This is supervised by the EF label itself (dense), and does
                    # NOT smooth away the genuine multi-beat oscillation.
                    ef_from_area = soft_amplitude_ef(area_seq_full, beta=amp_beta)  # (B,)
                    loss_amp = torch.nn.functional.mse_loss(ef_from_area, ef_true.view(-1))

                    # light smoothness: only suppresses single-bin spikes, not beats
                    diff = area_seq_full[:, 1:] - area_seq_full[:, :-1]
                    loss_smooth = (diff ** 2).mean()

                    # weak absolute-scale anchor: mean predicted area ~ a sane positive
                    # magnitude (keeps the curve in physical-ish units, avoids collapse).
                    # We anchor to the GT area scale if available, else to ~1.0 (=1000 px).
                    if area_tgt is not None:
                        valid = area_tgt[:, 4]
                        gt_mean_area = ((area_tgt[:, 0] + area_tgt[:, 1]) / 2.0) / 1000.0
                        pred_mean_area = area_seq_full.mean(dim=1)
                        loss_anchor = (((pred_mean_area - gt_mean_area) ** 2) * valid).sum() / (valid.sum() + 1e-6)
                    else:
                        loss_anchor = ((area_seq_full.mean(dim=1) - 1.0) ** 2).mean()

                    loss = loss_ef + l_smooth * loss_smooth + l_anchor * loss_anchor
                    if amp_active:
                        loss = loss + l_amp * loss_amp

                    if log_area and (not printed_area):
                        seq0 = area_seq_full[0].detach().cpu().numpy()
                        efa = ef_from_area[0].item()
                        print(f"    [area] amp_active={amp_active} ef_loss={loss_ef.item():.2f} "
                              f"amp_loss={loss_amp.item():.2f} smooth={loss_smooth.item():.4f} "
                              f"anchor={loss_anchor.item():.3f} | EF_true={ef_true.view(-1)[0].item():.1f} "
                              f"EF_from_area={efa:.1f} | seq[0]={np.array2string(seq0, precision=2, max_line_width=200)}",
                              flush=True)
                        printed_area = True

                if train:
                    optim.zero_grad()
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optim.step()
                    if ema_callback is not None:
                        ema_callback(model)

                total += loss.item() * X.size(0)
                n += X.size(0)
                pbar.set_postfix_str("{:.2f} ({:.2f})".format(total / n, loss.item()))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)
    return total / n, yhat, y


if __name__ == "__main__":
    run()