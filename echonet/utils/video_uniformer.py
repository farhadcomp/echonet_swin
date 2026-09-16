"""Functions for training and running EF prediction with UniFormer-S.

This file is intentionally close to the original EchoNet video.py.
Main changes vs baseline:
  1. Supports UniFormer-S as the model (in addition to Swin3D variants).
  2. Replace model head with a regression head (Dropout + Linear).
  3. Resize 112x112 EchoNet clips to 224x224 before feeding the model.
  4. Use MSE loss for EF regression.

Fixes applied:
  FIX-1  Lower backbone LR (1e-5) vs head LR (5e-4) to protect pretrained weights.
  FIX-2  Warmup (5 epochs LinearLR) → CosineAnnealingLR (eta_min=1e-7), no hard zero.
  FIX-3  Train longer: default num_epochs raised to 45.
  FIX-4  Checkpoint on best val R² (not val loss) — more stable signal.
  FIX-5  Gradient clipping (max_norm=1.0) to avoid early spike instability.
  FIX-6  Replace BatchNorm with GroupNorm for DataParallel multi-GPU compatibility.
         UniFormer uses BN in its local MHRA blocks. DataParallel computes BN
         statistics per-GPU independently, corrupting val normalisation.
         GroupNorm computes per-sample statistics — no cross-GPU sync needed,
         fully compatible with DataParallel on any number of GPUs.
"""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import echonet

import copy

class ModelEMA:
    """Exponential moving average of model weights. Not an ensemble — a single
    averaged model that's more stable than the raw weights."""
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
                ema_sd[k].copy_(v)   # ints (e.g. num_batches_tracked) copied as-is


def replace_bn_with_gn(module, num_groups=32):
    """Recursively replace all BatchNorm layers with GroupNorm.

    GroupNorm computes statistics per sample within channel groups,
    so it does not require cross-GPU synchronisation. This makes it
    fully compatible with DataParallel on multiple GPUs, unlike BatchNorm
    which needs SyncBatchNorm (only supported by DistributedDataParallel).

    Args:
        module:     The PyTorch module to convert in-place.
        num_groups: Number of channel groups for GroupNorm. 32 is standard.
                    Automatically reduced if num_channels < num_groups.
    """
    for name, child in module.named_children():
        if isinstance(
            child,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        ):
            num_channels = child.num_features
            # Ensure num_channels is divisible by groups
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1
            gn = torch.nn.GroupNorm(
                num_groups=groups,
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine,
            )
            # Copy learned affine parameters (weight and bias) if present
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            # Recurse into child modules
            replace_bn_with_gn(child, num_groups)


@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="EF")
@click.option("--model_name", type=str, default="uniformer_s")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--uniformer_weights", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to pretrained UniFormer .pth file from HuggingFace.")
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
@click.option("--backbone_lr", type=float, default=1e-5,
              help="LR for pretrained backbone layers.")
@click.option("--head_lr", type=float, default=5e-4,
              help="LR for the regression head.")
@click.option("--warmup_epochs", type=int, default=5,
              help="Number of linear-warmup epochs before cosine annealing kicks in.")
@click.option("--grad_clip", type=float, default=1.0,
              help="Max gradient norm for clipping. Set 0 to disable.")
@click.option("--mask_source", type=str, default="gt",
              help="'gt' for ground truth tracings, 'predicted' for MaskedVideos folder.")
@click.option("--mask_dir", type=str, default="MaskedVideos",
              help="Folder name inside data_dir containing predicted mask videos.")
def run(
    data_dir=None,
    output=None,
    task="EF",
    model_name="uniformer_s",
    pretrained=True,
    weights=None,
    uniformer_weights=None,
    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    frames=36,
    period=4,
    num_train_patients=None,
    num_workers=4,
    batch_size=16,
    device=None,
    seed=0,
    backbone_lr=1e-5,
    head_lr=5e-4,
    warmup_epochs=5,
    grad_clip=1.0,
    mask_source="gt",
    mask_dir="MaskedVideos"
):
    """Trains/tests EF prediction model."""

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Output directory
    if output is None:
        output = os.path.join(
            "output", "video",
            "{}_{}_{}_{}".format(
                model_name, frames, period,
                "pretrained" if pretrained else "random",
            ),
        )
    os.makedirs(output, exist_ok=True)

    # Device
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

        # model = uniformer_small()
        model = uniformer_small(in_chans=4)


        # if pretrained and uniformer_weights is not None:
        #     checkpoint = torch.load(
        #         uniformer_weights, map_location="cpu", weights_only=False
        #     )
        #     state_dict = checkpoint.get("model", checkpoint)
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
        #     missing, unexpected = model.load_state_dict(state_dict, strict=False)
        #     print(f"  Pretrained weights loaded.", flush=True)
        #     print(f"  Missing keys (expected — head): {missing}", flush=True)
        #     print(f"  Unexpected keys: {unexpected}", flush=True)

        if pretrained and uniformer_weights is not None:
            checkpoint = torch.load(
                uniformer_weights, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint.get("model", checkpoint)
            # Remove head — we replace it
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}

            # ---------------------------------------------------------- #
            # patch_embed1.proj.weight: pretrained shape is [64, 3, 3, 4, 4]
            # Our model needs                               [64, 4, 3, 4, 4]
            # Initialise the 4th (mask) channel by averaging the 3 RGB
            # channels — principled warm start, avoids random init spike.
            # ---------------------------------------------------------- #
            pe_key = "patch_embed1.proj.weight"
            if pe_key in state_dict:
                w = state_dict[pe_key]                    # [64, 3, 3, 4, 4]
                w_extra = w.mean(dim=1, keepdim=True)     # [64, 1, 3, 4, 4]
                state_dict[pe_key] = torch.cat([w, w_extra], dim=1)  # [64, 4, 3, 4, 4]
                print("  patch_embed1.proj: extended 3→4 channels (mask channel init by RGB mean).", flush=True)

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Pretrained weights loaded.", flush=True)
            print(f"  Missing keys (expected — head only): {missing}", flush=True)
            print(f"  Unexpected keys: {unexpected}", flush=True)
        elif pretrained and uniformer_weights is None:
            print("WARNING: --pretrained set but --uniformer_weights not provided.", flush=True)

        # Regression head
        model.head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 1),
        )
        model.head[1].bias.data[0] = 55.6

        # FIX-6: BatchNorm → GroupNorm for DataParallel multi-GPU safety
        replace_bn_with_gn(model, num_groups=32)
        print("  BatchNorm → GroupNorm conversion done.", flush=True)

    elif model_name in ("swin3d_s", "swin3d_t"):
        # Swin3D uses LayerNorm — safe with DataParallel already
        print(f"Initializing {model_name.upper()} for EF regression", flush=True)
        pretrained_weights = "KINETICS400_V1" if pretrained else None
        if model_name == "swin3d_t":
            model = torchvision.models.video.swin3d_t(weights=pretrained_weights)
        else:
            model = torchvision.models.video.swin3d_s(weights=pretrained_weights)
        in_features = model.head.in_features
        model.head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features, 1),
        )
        model.head[1].bias.data[0] = 55.6

    else:
        print(f"Initializing original video model: {model_name}", flush=True)
        model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.fc.bias.data[0] = 55.6

    # DataParallel — safe for all models now that UniFormer uses GroupNorm
    # if device.type == "cuda" and model_name != "uniformer_s":
    #     model = torch.nn.DataParallel(model)
    # model.to(device)

    # For UniFormer-S: single GPU, no DataParallel (GroupNorm handles normalisation)
# For all other models: DataParallel across available GPUs
    if model_name == "uniformer_s":
        model = model.to(device)
        print(f"  UniFormer-S on single GPU: {device}", flush=True)
        ema = ModelEMA(model, decay=0.999)
        print("  EMA initialized (decay=0.999)", flush=True)
    else:
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
        model = model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded weights from {weights}", flush=True)

    # ------------------------------------------------------------------ #
    # FIX-1: Differential learning rates
    # ------------------------------------------------------------------ #
    print(
        f"Optimizer: AdamW | backbone_lr={backbone_lr} | head_lr={head_lr} | "
        f"weight_decay={weight_decay}",
        flush=True,
    )

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optim = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    # ------------------------------------------------------------------ #
    # FIX-2: Warmup + Cosine annealing
    # ------------------------------------------------------------------ #
    cosine_epochs = max(1, num_epochs - warmup_epochs)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cosine_epochs, eta_min=1e-7,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )
    print(
        f"Scheduler: LinearWarmup({warmup_epochs} epochs) → "
        f"CosineAnnealing({cosine_epochs} epochs, eta_min=1e-7)",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=data_dir, split="train", add_mask=False)
    )
    kwargs = {
        "target_type": task,
        "mean": mean,
        "std": std,
        "length": frames,
        "period": period,
        "add_mask": True,
        "mask_source": mask_source,
        "mask_dir": mask_dir,
    }
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    dataset["val"]   = echonet.datasets.Echo(root=data_dir, split="val",   **kwargs)

    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    with open(os.path.join(output, "log.csv"), "a") as f:

        epoch_resume = 0
        bestR2 = -float("inf")

        try:
            checkpoint = torch.load(
                os.path.join(output, "checkpoint.pt"),
                map_location="cpu", weights_only=False,
            )
            model.load_state_dict(checkpoint["state_dict"])
            optim.load_state_dict(checkpoint["opt_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            bestR2 = checkpoint.get("best_r2", -float("inf"))
            f.write("Resuming from epoch {}\n".format(epoch_resume))
            f.flush()
            print(f"Resuming from epoch {epoch_resume}, best val R²={bestR2:.4f}", flush=True)
        except FileNotFoundError:
            f.write("Starting run from scratch\n")
            f.flush()
            print("Starting run from scratch", flush=True)

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)

            for phase in ["train", "val"]:
                start_time = time.time()

                # if device.type == "cuda":
                #     for i in range(torch.cuda.device_count()):
                #         torch.cuda.reset_peak_memory_stats(i)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=(phase == "train"),
                    pin_memory=(device.type == "cuda"),
                    drop_last=(phase == "train"),
                )

                loss, yhat, y = run_epoch(
                    model, dataloader, phase == "train",
                    optim, device, model_name=model_name, grad_clip=grad_clip,
                    ema_callback=(ema.update if phase == "train" else None),
                )

                # if device.type == "cuda":
                #     max_allocated = sum(
                #         torch.cuda.max_memory_allocated(i)
                #         for i in range(torch.cuda.device_count())
                #     )
                #     max_reserved = sum(
                #         torch.cuda.max_memory_reserved(i)
                #         for i in range(torch.cuda.device_count())
                #     )
                if device.type == "cuda":
                    max_allocated = torch.cuda.max_memory_allocated(device)
                    max_reserved  = torch.cuda.max_memory_reserved(device)
                else:
                    max_allocated = 0
                    max_reserved = 0

                epoch_r2 = sklearn.metrics.r2_score(y, yhat)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(
                    epoch, phase, loss, epoch_r2,
                    time.time() - start_time, y.size,
                    max_allocated, max_reserved, batch_size,
                ))
                f.flush()

            scheduler.step()
            current_lrs = [pg["lr"] for pg in optim.param_groups]
            print(
                f"  LRs after step: backbone={current_lrs[0]:.2e}, head={current_lrs[1]:.2e}",
                flush=True,
            )

            # ---- Evaluate the EMA model on val ----
            val_loader_ema = torch.utils.data.DataLoader(
                dataset["val"], batch_size=batch_size, num_workers=num_workers,
                shuffle=False, pin_memory=(device.type == "cuda"),
            )
            _, yhat_ema, y_ema = run_epoch(
                ema.ema, val_loader_ema, False, None, device,
                model_name=model_name, grad_clip=0,
            )
            ema_val_r2 = sklearn.metrics.r2_score(y_ema, yhat_ema)
            raw_val_r2 = epoch_r2
            print(f"  raw val R²={raw_val_r2:.4f} | EMA val R²={ema_val_r2:.4f}", flush=True)

            # pick whichever is better for checkpointing
            if ema_val_r2 >= raw_val_r2:
                val_r2 = ema_val_r2
                best_state = ema.ema.state_dict()
                tag = "EMA"
            else:
                val_r2 = raw_val_r2
                best_state = model.state_dict()
                tag = "raw"

            save = {
                "epoch": epoch,
                "state_dict": best_state,                 # EMA or raw, whichever won
                "raw_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict(),
                "period": period,
                "frames": frames,
                "best_r2": bestR2,
                "loss": loss,
                "r2": val_r2,
                "opt_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))

            if val_r2 > bestR2:
                torch.save(save, os.path.join(output, "best.pt"))
                bestR2 = val_r2
                print(
                    f"  ✓ New best val R²={bestR2:.4f} ({tag}) at epoch {epoch} — saved best.pt",
                    flush=True,
                )

        if num_epochs != 0:
            checkpoint = torch.load(
                os.path.join(output, "best.pt"),
                map_location="cpu", weights_only=False,
            )
            model.load_state_dict(checkpoint["state_dict"])
            f.write("Best val R² {:.4f} from epoch {}\n".format(
                checkpoint["r2"], checkpoint["epoch"],
            ))
            f.flush()
            print(
                f"Loaded best.pt: val R²={checkpoint['r2']:.4f} from epoch {checkpoint['epoch']}",
                flush=True,
            )

        # ------------------------------------------------------------------ #
        # Test / val evaluation
        # ------------------------------------------------------------------ #
        if run_test:
            for split in ["val", "test"]:

                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"),
                )
                loss, yhat, y = run_epoch(
                    model, dataloader, False, None, device,
                    model_name=model_name, grad_clip=0,
                )

                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score),
                ))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error),
                ))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *tuple(map(math.sqrt, echonet.utils.bootstrap(
                        y, yhat, sklearn.metrics.mean_squared_error
                    ))),
                ))
                f.flush()

                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers,
                    shuffle=False, pin_memory=(device.type == "cuda"),
                )
                loss, yhat, y = run_epoch(
                    model, dataloader, False, None, device,
                    save_all=True, block_size=batch_size,
                    model_name=model_name, grad_clip=0,
                )
                yhat_mean = np.array(list(map(lambda x: x.mean(), yhat)))

                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat_mean, sklearn.metrics.r2_score),
                ))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *echonet.utils.bootstrap(y, yhat_mean, sklearn.metrics.mean_absolute_error),
                ))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(
                    split, *tuple(map(math.sqrt, echonet.utils.bootstrap(
                        y, yhat_mean, sklearn.metrics.mean_squared_error,
                    ))),
                ))
                f.flush()

                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for filename, pred in zip(ds.fnames, yhat):
                        for i, p in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))

                echonet.utils.latexify()

                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat_mean.min())
                upper = max(y.max(), yhat_mean.max())
                plt.scatter(y, yhat_mean, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat_mean)
                    auc_score = sklearn.metrics.roc_auc_score(y > thresh, yhat_mean)
                    print(thresh, auc_score)
                    plt.plot(fpr, tpr)
                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)


def run_epoch(
    model,
    dataloader,
    train,
    optim,
    device,
    save_all=False,
    block_size=None,
    model_name="uniformer_s",
    grad_clip=1.0,
    ema_callback=None,
):
    """Run one epoch of training/evaluation for EF prediction."""

    model.train(train)

    total = 0.0
    n = 0
    s1 = 0.0
    s2 = 0.0
    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for X, outcome in dataloader:

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                # Handle all-clips mode: (batch, n_clips, C, F, H, W)
                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)
                elif len(X.shape) == 5:
                    pass
                else:
                    raise ValueError(f"Unexpected input shape: {X.shape}")

                # Upsample EchoNet 112×112 → 224×224
                if model_name in ("uniformer_s", "swin3d_s", "swin3d_t"):
                    X = torch.nn.functional.interpolate(
                        X,
                        size=(X.shape[2], 224, 224),
                        mode="trilinear",
                        align_corners=False,
                    )

                s1 += outcome.sum().item()
                s2 += (outcome ** 2).sum().item()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([
                        model(X[j : j + block_size, ...])
                        for j in range(0, X.shape[0], block_size)
                    ])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome.float())

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

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(
                    total / n, loss.item(), s2 / n - (s1 / n) ** 2,
                ))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y


if __name__ == "__main__":
    run()