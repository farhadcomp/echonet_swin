import torch
import torchvision
import echonet
import matplotlib.pyplot as plt

# 1. Rebuild the MC Dropout Architecture
model = torchvision.models.video.swin3d_s(weights=None)
in_features = model.head.in_features
model.head = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(in_features, 1)
)

# 2. Load your new weights (with DataParallel wrap)
checkpoint = torch.load("output/video/swin_2gpu_test/best.pt", map_location="cuda")
model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()

# 3. The MC Dropout Hack: Put in eval mode, but wake up Dropout
model.eval()
for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()

# 4. Load Test Dataset
ds = echonet.datasets.Echo(split="test")
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

# 5. Run the Monte Carlo Simulation for 15 patients
MC_SAMPLES = 50
NUM_PATIENTS = 50

targets = []
means = []
stds = []

print(f"Running evaluation for {NUM_PATIENTS} patients with {MC_SAMPLES} MC passes each...\n")

with torch.no_grad():
    for idx, (video, target) in enumerate(dataloader):
        if idx >= NUM_PATIENTS:
            break
        
        video = video.cuda()
        target_val = target.item()
        
        # Run the video through the model MC_SAMPLES times
        predictions = []
        for _ in range(MC_SAMPLES):
            pred = model(video)
            predictions.append(pred.item())
            
        # Calculate Mean and Std Dev
        tensor_preds = torch.tensor(predictions)
        mean_ef = tensor_preds.mean().item()
        uncertainty = tensor_preds.std().item()
        
        targets.append(target_val)
        means.append(mean_ef)
        stds.append(uncertainty)
        
        print(f"Patient {idx+1:02d} | True: {target_val:5.2f}% | Pred: {mean_ef:5.2f}% ± {uncertainty:4.2f}%")

# 6. Plotting the results
print("\nGenerating scatter plot with error bars...")
plt.figure(figsize=(10, 8))

# Plot the scatter with error bars
plt.errorbar(targets, means, yerr=stds, fmt='o', capsize=5, 
             ecolor='red', markerfacecolor='royalblue', markeredgecolor='darkblue', 
             alpha=0.8, label='MC Dropout Predictions')

# Add a perfect prediction diagonal line (y = x)
min_val = min(min(targets), min(means)) - 10
max_val = max(max(targets), max(means)) + 10
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

# Formatting the plot
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('True Ejection Fraction (%)', fontsize=12)
plt.ylabel('Predicted Ejection Fraction (%)', fontsize=12)
plt.title(f'({NUM_PATIENTS} Patients, {MC_SAMPLES} Passes/Patient)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')

# Save the plot
plot_path = "mc_dropout_scatter.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to: {plot_path}")