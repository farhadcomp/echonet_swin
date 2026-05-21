"""Functions for training and running Swin-Unet segmentation."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import monai  # <--- NEW: Medical Open Network for AI

import echonet

# --- THE SWIN-UNET WRAPPER ---
class SwinUnetWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the MONAI Swin-Unet (SwinUNETR) in 2D mode
        self.model = monai.networks.nets.SwinUNETR(
            # img_size=(224, 224), # Swin mathematically requires 224x224
            in_channels=3,       # 3 RGB frames
            out_channels=1,      # 1 channel for binary mask (Heart / Not Heart)
            spatial_dims=2       # 2D Segmentation
        )

    def forward(self, x):
        # 1. Upscale the 112x112 EchoNet frame to 224x224 for the Transformer
        x_224 = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 2. Pass through Swin-Unet
        logits_224 = self.model(x_224)
        
        # 3. Downscale back to 112x112 so the downstream loss functions don't crash
        logits_112 = torch.nn.functional.interpolate(logits_224, size=(112, 112), mode='bilinear', align_corners=False)
        
        # 4. Package it in the dictionary format that the Stanford code expects
        return {"out": logits_112}
# -----------------------------------------------

@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=str, default="swinunet") # <--- CHANGED: Allows custom string
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--save_video/--skip_video", default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-4) # <--- CHANGED: Transformers like a slightly higher learning rate
@click.option("--weight_decay", type=float, default=0.05) # <--- CHANGED: Crucial for Transformers
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=16) # <--- CHANGED: Lowered slightly for Swin-Unet VRAM
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    model_name="swinunet",
    pretrained=False,
    weights=None,
    run_test=False,
    save_video=False,
    num_epochs=50,
    lr=1e-4,
    weight_decay=0.05,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=4,
    batch_size=16,
    device=None,
    seed=0,
):
    """Trains/tests segmentation model."""

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 🚨 INIT MODEL INTERCEPT 🚨 ---
    if model_name.lower() == "swinunet":
        print("Initializing Pure Transformer Pipeline: Swin-Unet...")
        model = SwinUnetWrapper()
    else:
        print(f"Initializing standard torchvision model: {model_name}...")
        model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=False)
        model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)
    # ----------------------------------

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer (Using AdamW for Transformers if swinunet is selected)
    if model_name.lower() == "swinunet":
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs)
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
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                # Using local run_epoch function
                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, phase == "train", optim, device)
                
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    epoch, phase, loss, overall_dice, large_dice, small_dice,
                    time.time() - start_time, large_inter.size,
                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                    sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            for split in ["val", "test"]:
                dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, False, None, device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(large_inter, large_union, echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(small_inter, small_union, echonet.utils.dice_similarity_coefficient)))
                f.flush()

    # Saving videos with segmentations
    dataset = echonet.datasets.Echo(root=data_dir, split="test",
                                    target_type=["Filename", "LargeIndex", "SmallIndex"], 
                                    mean=mean, std=std,
                                    length=None, max_length=None, period=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        model.eval()
        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    y = np.concatenate([model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])
                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)
                        f, c, h, w = video.shape
                        video = np.concatenate((video, video), 3)
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])
                        video = np.concatenate((video, np.zeros_like(video)), 2)
                        size = (logit > 0).sum((1, 2))
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        for (frame_idx, s) in enumerate(size):
                            video[:, :, int(round(115 + 100 * s)), int(round(frame_idx / len(size) * 200 + 10))] = 255.
                            if frame_idx in systole:
                                video[:, :, 115:224, int(round(frame_idx / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on; x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            if frame_idx == large_index[i]:
                                video[:, :, d, int(round(frame_idx / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if frame_idx == small_index[i]:
                                video[:, :, d, int(round(frame_idx / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            r, c_idx = skimage.draw.disk((int(round(115 + 100 * s)), int(round(frame_idx / len(size) * 200 + 10))), 4.1)
                            video[frame_idx, :, r, c_idx] = 255.

                        video = video.transpose(1, 0, 2, 3).astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)
                        start += offset

def run_epoch(model, dataloader, train, optim, device):
    total, n, pos, neg, pos_pix, neg_pix = 0., 0, 0, 0, 0, 0
    model.train(train)
    large_inter, large_union, small_inter, small_union = 0, 0, 0, 0
    large_inter_list, large_union_list, small_inter_list, small_union_list = [], [], [], []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                pos += (large_trace == 1).sum().item() + (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item() + (small_trace == 0).sum().item()
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy() + (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy() + (small_trace == 0).sum(0).to("cpu").detach().numpy()

                large_frame, large_trace = large_frame.to(device), large_trace.to(device)
                y_large = model(large_frame)["out"]
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                small_frame, small_trace = small_frame.to(device), small_trace.to(device)
                y_small = model(small_frame)["out"]
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item()
                n += large_trace.size(0)
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(
                    total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, 
                    -p * math.log(p) - (1 - p) * math.log(1 - p), 
                    (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 
                    2 * large_inter / (large_union + large_inter), 
                    2 * small_inter / (small_union + small_inter)))
                pbar.update()

    return (total / n / 112 / 112, np.array(large_inter_list), np.array(large_union_list), np.array(small_inter_list), np.array(small_union_list))

def _video_collate_fn(x):
    video, target = zip(*x)
    i = list(map(lambda t: t.shape[1], video))
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))
    target = zip(*target)
    return video, target, i

if __name__ == '__main__':
    run()