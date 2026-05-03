import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import get_cifar10h_dataloaders
from evaluate_auto import entropy

def save_entropy_splits(root_dir):
    _, _, test_dl, all_probs = get_cifar10h_dataloaders(root_dir, batch_size=128, num_workers=0)
    
    # We know test indices are indices[8000:10000] from data.py
    # Re-generate it to be safe and match the DataLoader structure
    np.random.seed(42)
    indices = np.arange(10000)
    np.random.shuffle(indices)
    test_idx = indices[8000:]
    
    test_probs = all_probs[test_idx]
    test_entropies = entropy(test_probs)
    
    # Get argsort
    sorted_idx = np.argsort(test_entropies)
    
    lowest_entropy_relative_idx = sorted_idx[:50]
    highest_entropy_relative_idx = sorted_idx[-50:]
    
    # Save mapping for FGSM / Grad-CAM scripts
    os.makedirs(os.path.join(root_dir, 'attacks'), exist_ok=True)
    np.save(os.path.join(root_dir, 'attacks', 'lowest_entropy_indices.npy'), lowest_entropy_relative_idx)
    np.save(os.path.join(root_dir, 'attacks', 'highest_entropy_indices.npy'), highest_entropy_relative_idx)
    
    print("Found Top 50 High Entropy and Bottom 50 Low Entropy test samples.")
    print("Saved relative test indices to attacks/")
    
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_entropy_splits(root)
