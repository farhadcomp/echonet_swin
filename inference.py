import torch
import torchvision
import numpy as np
import cv2
import echonet

# 1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./output/segmentation/deeplabv3_resnet50_random/best.pt"
VIDEO_PATH = "../../echo/Videos/0X73E1750D79C18CE6.avi" # Change to any video filename
OUTPUT_PATH = "prediction_overlay.avi"

def run_inference():
    # 2. Load the Model Architecture
    print(f"loading model from {MODEL_PATH}...")
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    # --- UPDATE THIS SECTION IN inference.py ---

    # Load the trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Create a new dictionary without the 'module.' prefix
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove 'module.' prefix
        new_state_dict[name] = v

    # Now load the cleaned state_dict
    model.load_state_dict(new_state_dict)

    model.to(DEVICE)
    model.eval()

    # -------------------------------------------
    # 3. Load Video using OpenCV
    cap = cv2.VideoCapture(VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Define Video Writer to save results
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"Processing video: {VIDEO_PATH}")
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame for the model (Resize to 112x112 as per EchoNet)
            img = cv2.resize(frame, (112, 112))
            img = img.transpose((2, 0, 1)) / 255.0 # HWC to CHW
            img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0).to(DEVICE)

            # Inference
            output = model(img_tensor)["out"]
            prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
            mask = (prediction > 0.5).astype(np.uint8) # Binary threshold

            # Resize mask back to original video size
            mask_resized = cv2.resize(mask, (width, height))

            # Overlay mask on original frame (Red color)
            overlay = frame.copy()
            overlay[mask_resized > 0] = [0, 0, 255] # BGR for Red
            
            # Blend original and overlay
            combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            out.write(combined)

    cap.release()
    out.release()
    print(f"✅ Finished! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_inference()
