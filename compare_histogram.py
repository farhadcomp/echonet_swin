import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Import the Stanford codebase to use their exact video loader
import echonet

# 1. Setup the paths (replace with an actual patient ID!)
base_dir = "/home/AD.UNLV.EDU/farhadik/echo"
filename = "0X1002E8FBACD08477"  # <--- REPLACE THIS WITH A REAL ID

avi_path = os.path.join(base_dir, "Videos", f"{filename}.avi")
pt_path = os.path.join(base_dir, "Tensors", f"{filename}.pt")

print(f"Loading original AVI from: {avi_path}")
# Load AVI exactly how the original pipeline did
video_avi = echonet.utils.loadvideo(avi_path).astype(np.float32)

print(f"Loading your Tensor from: {pt_path}")
# Load your PyTorch tensor (before you apply the / 255.0 normalization)
video_pt = torch.load(pt_path).float().numpy()

# 2. Flatten both into 1D arrays for the histogram
pixels_avi = video_avi.ravel()
pixels_pt = video_pt.ravel()

# 3. Check if they are mathematically identical
difference = np.abs(pixels_avi - pixels_pt).max()
print(f"Maximum pixel value difference between files: {difference}")

# 4. Plotting
plt.figure(figsize=(12, 5))

# --- Plot 1: AVI Original ---
plt.subplot(1, 2, 1)
plt.hist(pixels_avi, bins=50, color='red', alpha=0.7)
plt.title(f"Original .avi File\nMax val: {pixels_avi.max():.1f}")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.yscale('log')

# --- Plot 2: Your .pt Tensor ---
plt.subplot(1, 2, 2)
plt.hist(pixels_pt, bins=50, color='blue', alpha=0.7)
plt.title(f"Your .pt Tensor\nMax val: {pixels_pt.max():.1f}")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.yscale('log')

plt.tight_layout()
plt.savefig("./histogram/histogram_comparison.png", dpi=300)
print("Successfully saved plot to 'histogram_comparison.png'")
