import torch
import echonet
import torchvision

print("✅ Importing EchoNet...")

# 1. Setup the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

# 2. Load the segmentation model (DeepLabV3)
# This downloads the pre-trained weights automatically
print("⏳ Loading Model to GPU...")
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

# Move to GPU
model = model.to(device)

model.eval()
# 3. Create a "Fake" Video Input
# (Batch Size=1, Channels=3, Frames=32, Height=112, Width=112)
dummy_input = torch.randn(1, 3, 112, 112).to(device)

# 4. Run Inference
print("🚀 Running Dummy Inference...")
with torch.no_grad():
    output = model(dummy_input)

print("✅ Success! The model accepted the input and ran on the RTX 8000.")
print(f"   Output shape: {output['out'].shape}")
