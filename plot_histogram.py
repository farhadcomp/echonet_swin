import torch
import matplotlib.pyplot as plt
import numpy as np

import os
os.makedirs("./histogram", exist_ok=True)
# 1. Point this to ONE specific patient tensor on your drive
# Make sure to put a real filename here!
tensor_path = "/home/AD.UNLV.EDU/farhadik/echo/Tensors/0X1002E8FBACD08477.pt" 

print(f"Loading tensor from: {tensor_path}")
video_tensor = torch.load(tensor_path).float()

# Print the shape to confirm (Should be [Channels, Frames, Height, Width])
print(f"Video shape: {video_tensor.shape}")

# 2. Extract a single frame (Channel 0, Frame 0)
# We convert it to numpy and flatten it to 1D for the histogram
single_frame = video_tensor[0, 0, :, :].numpy()
pixels_single = single_frame.ravel()

# 3. Extract all frames
pixels_all = video_tensor.numpy().ravel()

# 4. Plotting
plt.figure(figsize=(12, 5))

# --- Plot 1: Single Frame ---
plt.subplot(1, 2, 1)
plt.hist(pixels_single, bins=50, color='blue', alpha=0.7)
plt.title("Pixel Distribution (Single Frame)")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency (Number of Pixels)")
plt.yscale('log') # Log scale helps see the smaller mid-gray values!

# --- Plot 2: Entire Video ---
plt.subplot(1, 2, 2)
plt.hist(pixels_all, bins=50, color='green', alpha=0.7)
plt.title("Pixel Distribution (All Frames)")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency (Number of Pixels)")
plt.yscale('log') 

plt.tight_layout()
plt.savefig("./histogram/noise_histogram.png", dpi=300)
print("Successfully saved plot to 'noise_histogram.png'")
