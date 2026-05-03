import os
import torch
from robustness import fgsm_attack
from data import get_cifar10h_dataloaders

# Minimal script to reproduce FGSM clamping bug
def run_test():
    root = os.path.dirname(os.path.abspath(__file__))
    # Adjust path assuming this script is in src/
    project_root = os.path.dirname(root)
    
    # Load normalized images 
    train_dl, _, _, _ = get_cifar10h_dataloaders(project_root, batch_size=4, num_workers=0)
    for images, targets in train_dl:
        print(f"[PRE-ATTACK] Original Min: {images.min().item():.4f}, Max: {images.max().item():.4f}")
        
        # Simulate an adversarial gradient
        data_grad = torch.ones_like(images)
        
        # Attack with epsilon
        perturbed = fgsm_attack(images, 0.05, data_grad)
        
        print(f"[POST-ATTACK] Perturbed Min: {perturbed.min().item():.4f}, Max: {perturbed.max().item():.4f}")
        
        # Check clipping damage
        if perturbed.max().item() <= 1.0 and images.max().item() > 1.0:
            print(">>> BUG REPRODUCED: Perturbed image was aggressively destroyed by hard [0, 1] clipping on normalized space!")
        break

if __name__ == "__main__":
    run_test()
