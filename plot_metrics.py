import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
# Read the target folder from the bash command line arguments. 
# If none is provided, it defaults to the current directory ("./")
TARGET_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "./"
LOG_FILE = os.path.join(TARGET_FOLDER, "log.csv")

def generate_plots():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Could not find {LOG_FILE}.")
        return

    epochs = []
    train_loss = []
    val_loss = []

    print(f"Parsing metrics from {LOG_FILE}...")
    
    # Manually parse to avoid crashing on weird text lines
    with open(LOG_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            
            # Only process lines that start with a number (the epoch)
            if len(parts) > 3 and parts[0].isdigit():
                epoch = int(parts[0])
                phase = parts[1]
                loss = float(parts[2])
                
                if phase == 'train':
                    epochs.append(epoch)
                    train_loss.append(loss)
                elif phase == 'val':
                    val_loss.append(loss)

    # Create a beautiful, publication-style figure
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_loss, label='Training Loss', color='royalblue', linewidth=2.5, marker='o', markersize=5)
    plt.plot(epochs, val_loss, label='Validation Loss', color='crimson', linewidth=2.5, marker='s', markersize=5)

    # Formatting the plot
    plt.title('Swin Transformer: Training vs. Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Smooth L1 Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the high-resolution image
    save_path = os.path.join(TARGET_FOLDER, "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Success! Plot saved to: {save_path}")

if __name__ == "__main__":
    generate_plots()