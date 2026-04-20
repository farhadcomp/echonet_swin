import torch
import torch.nn.functional as F
import torchvision.io
import echonet

# 1. Load a real patient video
print("Loading patient video from dataset...")
ds = echonet.datasets.Echo(split="test")
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
video, target = next(iter(dataloader))

# 2. Interpolate to 224x224
print("Interpolating video to 224x224...")
video_resized = F.interpolate(
    video, 
    size=(video.shape[2], 224, 224), 
    mode='trilinear', 
    align_corners=False
)

# 3. Format function: Convert from PyTorch math to standard video format
def format_for_saving(vid_tensor):
    # Remove the batch dimension: [1, C, T, H, W] -> [C, T, H, W]
    vid = vid_tensor.squeeze(0)
    
    # Normalize the pixel values to be between 0 and 255
    vid = vid - vid.min()
    vid = vid / vid.max()
    vid = (vid * 255).to(torch.uint8)
    
    # Rearrange dimensions: [Channels, Time, Height, Width] -> [Time, Height, Width, Channels]
    vid = vid.permute(1, 2, 3, 0)
    return vid

# 4. Process both videos
orig_avi = format_for_saving(video)
resized_avi = format_for_saving(video_resized)

# 5. Save them to the hard drive (EchoNet is recorded at 50 FPS)
print("Saving original_112.avi...")
torchvision.io.write_video("original_112.avi", orig_avi, fps=10)

print("Saving interpolated_224.avi...")
torchvision.io.write_video("interpolated_224.avi", resized_avi, fps=10)

print(f"\nDone! Patient EF was {target.item():.2f}%")
