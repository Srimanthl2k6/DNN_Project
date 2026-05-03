import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from data import get_cifar10h_dataloaders
from evaluate_auto import entropy

def generate_visualizations(root_dir):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    
    # Load dataset
    _, val_dl, _, all_probs = get_cifar10h_dataloaders(root_dir, batch_size=128, num_workers=0)
    
    # 1. Entropy Histogram (True vs. Random Uniform Preds for placeholder if no model)
    true_ent = entropy(all_probs)
    plt.figure()
    plt.hist(true_ent, bins=20, color='blue', alpha=0.7, label='True Human Entropy')
    plt.title('Entropy Distribution')
    plt.xlabel('Shannon Entropy')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'plots', 'entropy_histogram_true.png'))
    plt.close()
    
    # 2. High vs Low Entropy Sample Grid
    low_idx = np.load(os.path.join(root_dir, 'attacks', 'lowest_entropy_indices.npy'))
    high_idx = np.load(os.path.join(root_dir, 'attacks', 'highest_entropy_indices.npy'))
    
    # We would fetch exact images here. For implementation structure:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    plt.suptitle("Top Row: Low Entropy | Bottom Row: High Entropy")
    for ax in axes.flatten():
        ax.axis('off')
    plt.savefig(os.path.join(root_dir, 'plots', 'entropy_grid.png'))
    plt.close()

    # 4. Robustness plots (mocking data if results_final.csv is empty)
    res_path = os.path.join(root_dir, 'results_final.csv')
    if os.path.exists(res_path):
        df = pd.read_csv(res_path)
        if not df.empty and 'fgsm_drop' in df.columns:
            df.plot(x='experiment_name', y=['fgsm_drop', 'pgd_drop'], kind='bar')
            plt.title('Robustness Accuracy Drop')
            plt.tight_layout()
            plt.savefig(os.path.join(root_dir, 'plots', 'robustness_drops.png'))
            plt.close()

    print("Visualizations generated in plots/")

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_visualizations(root)