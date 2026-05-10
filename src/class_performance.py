import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from data import CIFAR10_CLASSES, get_cifar10h_dataloaders
from evaluate_auto import entropy
from model import CustomResNet18


BEST_CHECKPOINT = 'Exp_KL_Random_Linear.pth'


def load_checkpoint_state_dict(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_best_checkpoint_model(root_dir, device):
    checkpoint_path = os.path.join(root_dir, 'checkpoints', BEST_CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Missing best checkpoint: {checkpoint_path}')

    model = CustomResNet18(head_type='linear', pretrain_strategy='random').to(device)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))
    model.eval()
    return model


def run_inference(model, dataloader, device):
    predictions = []
    targets = []

    with torch.no_grad():
        for images, batch_targets in dataloader:
            logits = model(images.to(device))
            predictions.append(F.softmax(logits, dim=-1).cpu().numpy())
            targets.append(batch_targets.cpu().numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def compute_class_metrics(predictions, targets):
    hard_targets = np.argmax(targets, axis=1)
    true_entropies = entropy(targets)
    predicted_entropies = entropy(predictions)

    mean_kls = []
    entropy_correlations = []
    for class_idx in range(len(CIFAR10_CLASSES)):
        class_mask = hard_targets == class_idx
        class_targets = targets[class_mask]
        class_predictions = predictions[class_mask]

        if not np.any(class_mask):
            mean_kls.append(0.0)
            entropy_correlations.append(0.0)
            continue

        kl_values = np.sum(
            class_targets * (np.log(class_targets + 1e-12) - np.log(class_predictions + 1e-12)),
            axis=1,
        )
        mean_kls.append(float(np.mean(kl_values)))

        class_true_entropy = true_entropies[class_mask]
        class_predicted_entropy = predicted_entropies[class_mask]
        if (
            len(class_true_entropy) < 2
            or np.std(class_true_entropy) == 0.0
            or np.std(class_predicted_entropy) == 0.0
        ):
            entropy_correlations.append(0.0)
        else:
            entropy_correlations.append(float(pearsonr(class_true_entropy, class_predicted_entropy)[0]))

    return mean_kls, entropy_correlations


def plot_class_performance(root_dir, mean_kls, entropy_correlations):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    positions = np.arange(len(CIFAR10_CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    axes[0].bar(positions, mean_kls, color='#4c78a8')
    axes[0].set_title('Per-Class Mean KL Divergence')
    axes[0].set_ylabel('Mean KL Divergence')
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(CIFAR10_CLASSES, rotation=35, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(positions, entropy_correlations, color='#f58518')
    axes[1].set_title('Per-Class Pearson Entropy Correlation')
    axes[1].set_ylabel('Pearson r')
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(CIFAR10_CLASSES, rotation=35, ha='right')
    axes[1].axhline(0.0, color='black', linewidth=1.0)
    axes[1].grid(axis='y', alpha=0.3)

    fig.tight_layout()
    output = os.path.join(root_dir, 'plots', 'class_performance.png')
    fig.savefig(output)
    plt.close(fig)
    return output


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_dl, _ = get_cifar10h_dataloaders(root_dir, batch_size=128, num_workers=0)
    model = load_best_checkpoint_model(root_dir, device)

    predictions, targets = run_inference(model, test_dl, device)
    mean_kls, entropy_correlations = compute_class_metrics(predictions, targets)
    output = plot_class_performance(root_dir, mean_kls, entropy_correlations)
    print(f'Saved class-conditional performance plot to {output}')


if __name__ == '__main__':
    main()