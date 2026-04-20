import os
import torch
import torchvision.io
from tqdm import tqdm

# --- CONFIGURATION ---
# Change this to the exact path of your EchoNet dataset!
ECHONET_FOLDER = "/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic" 
VIDEO_DIR = os.path.join(ECHONET_FOLDER, "Videos")
TENSOR_DIR = os.path.join(ECHONET_FOLDER, "Tensors")

# Create the output directory if it doesn't exist
os.makedirs(TENSOR_DIR, exist_ok=True)

def process_all_videos():
    # Get all .avi files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    print(f"Found {len(video_files)} videos to process. Starting conversion...\n")
    
    # Loop through with a progress bar
    for file_name in tqdm(video_files, desc="Converting to .pt tensors"):
        vid_path = os.path.join(VIDEO_DIR, file_name)
        tensor_path = os.path.join(TENSOR_DIR, file_name.replace('.avi', '.pt'))
        
        # Skip if we already processed this video (allows you to pause/resume)
        if os.path.exists(tensor_path):
            continue
            
        try:
            # 1. Read the video (Returns [Time, Height, Width, Channels] in uint8)
            # pts_unit="sec" is required by newer torchvision versions
            vid, audio, info = torchvision.io.read_video(vid_path, pts_unit="sec")
            
            # 2. Rearrange to PyTorch format: [Channels, Time, Height, Width]
            vid = vid.permute(3, 0, 1, 2)
            
            # 3. Save directly to SSD as a raw PyTorch tensor
            # Keeping it as uint8 saves massive amounts of disk space!
            torch.save(vid, tensor_path)
            
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")

if __name__ == "__main__":
    process_all_videos()
    print("\nDataset fully converted! Ready for high-speed training.")
