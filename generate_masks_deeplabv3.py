"""Generate MaskedVideos using official EchoNet DeepLabV3 segmentation weights."""

import os
import sys
import torch
import torchvision
import numpy as np
import tqdm

sys.path.insert(0, '/home/AD.UNLV.EDU/farhadik/tests/dynamic')
import echonet


def generate_masked_dataset(data_dir, weights_path, prob_threshold=0.5):
    print(f"Generating MaskedVideosEcho with DeepLabV3 (threshold={prob_threshold})")
    device = torch.device("cuda:0")

    output_dir = os.path.join(data_dir, "MaskedVideosEcho")
    os.makedirs(output_dir, exist_ok=True)

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=False,
        aux_loss=False,
    )
    model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=1)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    # Remove 'module.' prefix added by DataParallel during training
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("DeepLabV3 model loaded.")

    dummy_ds = echonet.datasets.Echo(root=data_dir, split="train")
    mean, std = echonet.utils.get_mean_and_std(dummy_ds)
    print(f"Mean: {mean}, Std: {std}")

    dataset = echonet.datasets.Echo(
        root=data_dir,
        split="all",
        target_type=["Filename"],
        period=1,
        length=None,
        max_length=None,
        mean=mean,
        std=std,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=4, shuffle=False
    )

    coverage_list = []

    print(f"Processing {len(dataset)} videos...")
    with torch.no_grad():
        for (x, target) in tqdm.tqdm(dataloader):
            video_frames = x[0].permute(1, 0, 2, 3).to(device)

            chunk_size = 32
            all_masks = []
            for i in range(0, video_frames.shape[0], chunk_size):
                chunk = video_frames[i:i+chunk_size]
                logits = model(chunk)["out"]
                probs = torch.sigmoid(logits)
                binary = (probs > prob_threshold).float()
                all_masks.append(binary.cpu())

            binary_mask = torch.cat(all_masks, dim=0)

            coverage = binary_mask.mean().item()
            coverage_list.append(coverage)

            original_frames = video_frames.cpu().numpy()
            original_frames *= std.reshape(1, 3, 1, 1)
            original_frames += mean.reshape(1, 3, 1, 1)
            original_frames = np.clip(original_frames, 0, 255)

            binary_mask_np = binary_mask.squeeze(1).numpy()
            masked_frames = original_frames * binary_mask_np[:, np.newaxis, :, :]
            masked_frames = masked_frames.astype(np.uint8)

            filename = target[0]
            echonet.utils.savevideo(
                os.path.join(output_dir, filename),
                masked_frames.transpose(1, 0, 2, 3),
                50
            )

    print(f"\nDone!")
    print(f"Mean coverage: {np.mean(coverage_list):.3f}")
    print(f"Min coverage:  {np.min(coverage_list):.3f}")
    print(f"Max coverage:  {np.max(coverage_list):.3f}")


if __name__ == "__main__":
    DATA_DIR = "/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic"
    WEIGHTS  = "/home/AD.UNLV.EDU/farhadik/tests/dynamic/deeplabv3_resnet50_random.pt"
    generate_masked_dataset(DATA_DIR, WEIGHTS, prob_threshold=0.5)
