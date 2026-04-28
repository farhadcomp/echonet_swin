"""Functions for training and running Baseline EF prediction for R(2+1)D."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torchvision
import tqdm

import echonet
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="EF")
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
    default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    task="EF",
    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,
    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
):
    """Trains/tests baseline EF prediction model using R(2+1)D."""

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- DDP Initialization ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # --------------------------

    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Standard R(2+1)D Architecture ---
    if local_rank == 0:
        print(f"Initializing Baseline {model_name} Architecture (Baseline EF)...")
        
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6   # Average EF

    model.to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if weights is not None:
        checkpoint = torch.load(weights, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])

    # --- CLASSIC CNN ENGINE (SGD + StepLR) ---
    if local_rank == 0:
        print("Engine: SGD + Momentum + StepLR (Classic Mode)", flush=True)
        
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
    # -----------------------------------------

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    
    # Baseline EF Only Kwargs
    kwargs = {"target_type": "EF",
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    # Set up datasets
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            if local_rank == 0:
                f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            if local_rank == 0:
                f.write("Starting run from scratch\n")
                f.flush()
        
        # --- EARLY STOPPING SETUP ---
        patience = 7  
        patience_counter = 0

        for epoch in range(epoch_resume, num_epochs):
            if local_rank == 0:
                print(f"Epoch #{epoch} | R(2+1)D Baseline EF Only", flush=True)
                
            for phase in ['train', 'val']:
                start_time = time.time()
                torch.cuda.reset_peak_memory_stats(local_rank)

                ds = dataset[phase]
                
                if phase == "train":
                    sampler = DistributedSampler(ds, shuffle=True)
                    sampler.set_epoch(epoch)
                    dataloader = torch.utils.data.DataLoader(
                        ds, batch_size=batch_size, num_workers=num_workers, 
                        sampler=sampler, pin_memory=True, drop_last=True
                    )
                else:
                    dataloader = torch.utils.data.DataLoader(
                        ds, batch_size=batch_size, num_workers=num_workers, 
                        shuffle=False, pin_memory=True, drop_last=False
                    )

                loss, yhat, y = run_epoch(model, dataloader, phase == "train", optim, device)
                
                if local_rank == 0:
                    f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                  phase,
                                                                  loss,
                                                                  sklearn.metrics.r2_score(y, yhat),
                                                                  time.time() - start_time,
                                                                  y.size,
                                                                  torch.cuda.max_memory_allocated(local_rank),
                                                                  torch.cuda.max_memory_reserved(local_rank),
                                                                  batch_size))
                    f.flush()

            # --- 🚨 DDP FIX 1: ONLY GPU 0 SAVES CHECKPOINTS 🚨 ---
            should_stop = torch.tensor(0).to(device) 

            if local_rank == 0:
                save = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'period': period,
                    'frames': frames,
                    'best_loss': bestLoss,
                    'loss': loss,
                    'r2': sklearn.metrics.r2_score(y, yhat),
                    'opt_dict': optim.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                }
                torch.save(save, os.path.join(output, "checkpoint.pt"))
                
                if loss < bestLoss:
                    torch.save(save, os.path.join(output, "best.pt"))
                    bestLoss = loss
                    patience_counter = 0 
                else:
                    patience_counter += 1
                    print("Early stopping counter: {} out of {}".format(patience_counter, patience), flush=True)
                    
                    if patience_counter >= patience:
                        print("Early stopping triggered! Exiting safely...", flush=True)
                        should_stop += 1 

            dist.broadcast(should_stop, src=0)
            if should_stop.item() == 1:
                break 
            
            dist.barrier()
            scheduler.step()
            # -------------------------------------------------------------

        # Load best weights
        if num_epochs != 0:
            dist.barrier() 
            checkpoint = torch.load(os.path.join(output, "best.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            if local_rank == 0:
                f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
                f.flush()

        # --- 🚨 DDP FIX 2: ISOLATE TESTING AND PLOTTING TO GPU 0 🚨 ---
        if run_test and local_rank == 0:
            for split in ["val", "test"]:
                # Performance without test-time augmentation
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                
                loss, yhat, y = run_epoch(model.module, dataloader, False, None, device)
                
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Performance with test-time augmentation
                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                
                loss, yhat, y = run_epoch(model.module, dataloader, False, None, device, save_all=True, block_size=batch_size)
                
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                echonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # --- Upgraded Scatter Plot with R2 Score ---
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)

                r2_val = sklearn.metrics.r2_score(y, yhat)
                plt.text(0.05, 0.95, f"$R^2$ = {r2_val:.3f}", 
                         transform=plt.gca().transAxes, fontsize=9, 
                         verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=4)

                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # --- Upgraded ROC Plot with Grid and Labels ---
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--", label="No Skill")
                
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    auc_score = sklearn.metrics.roc_auc_score(y > thresh, yhat)
                    print(f"Threshold {thresh}: {auc_score:.4f}")
                    plt.plot(fpr, tpr, label=f"EF < {thresh} (AUC: {auc_score:.2f})")

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.legend(loc="lower right", fontsize="x-small")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)

        # --- 🚨 DDP FIX 3: KEEP ALL GPUS ALIVE UNTIL THE VERY END 🚨 ---
        dist.barrier()
        if local_rank == 0:
            print("Evaluation and plotting complete! All GPUs shutting down safely.", flush=True)

def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    """Run one epoch of training/evaluation."""

    model.train(train)

    total = 0  
    n = 0      
    s1 = 0     
    s2 = 0     

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        with tqdm.tqdm(total=len(dataloader), disable=(local_rank != 0)) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.numpy())

                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                # Upscale 112x112 to 224x224 directly on GPU (matches your pre-trained weights)
                # X = torch.nn.functional.interpolate(X, size=(X.shape[2], 224, 224), mode='trilinear', align_corners=False)

                s1 += outcome.sum().item()
                s2 += (outcome ** 2).sum().item()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs[:, 0].to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs[:, 0].to("cpu").detach().numpy())

                # Pure EF Smooth L1 Loss
                loss = torch.nn.functional.smooth_l1_loss(outputs.view(-1), outcome, beta=1.0)
                
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y

if __name__ == '__main__':
    run()