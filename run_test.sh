python3 - <<'PY'
import torch

ckpt_path = "output/video/swin_2gpu_test/best.pt"  # change if needed
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("Checkpoint keys:")
print(ckpt.keys())

if "opt_dict" not in ckpt:
    print("\nNo opt_dict found in this checkpoint, so exact optimizer LR was not saved.")
else:
    print("\nOptimizer parameter groups:")
    for i, group in enumerate(ckpt["opt_dict"]["param_groups"]):
        print(f"Group {i}")
        print("  lr:", group.get("lr"))
        print("  initial_lr:", group.get("initial_lr", "not saved"))
        print("  weight_decay:", group.get("weight_decay"))
        print("  number of params:", len(group.get("params", [])))

if "scheduler_dict" in ckpt:
    print("\nScheduler state:")
    print(ckpt["scheduler_dict"])
PY