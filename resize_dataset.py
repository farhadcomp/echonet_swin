import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

def resize_tensors(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
    
    for fname in tqdm(files):
        # Load (Assuming shape: C, F, H, W)
        t = torch.load(os.path.join(input_dir, fname))
        
        # Interpolate to 224x224
        # We use bilinear for 2D spatial scaling within the frames
        t_resized = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
        
        # Save back to disk
        torch.save(t_resized.byte(), os.path.join(output_dir, fname))

resize_tensors("/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic/Tensors", "/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic/Tensors_224")

