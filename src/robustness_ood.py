import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from data import get_cifar10h_dataloaders
from losses import KLDivergenceLoss
from model import CustomResNet18


CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
BEST_CHECKPOINT = 'Exp_KL_Random_Linear.pth'
CORRUPTION_SPECS = {
    'gaussian_noise': [('mild', 0.05), ('moderate', 0.10), ('severe', 0.20)],
    'gaussian_blur': [('mild', 3), ('moderate', 5), ('severe', 7)],
    'contrast_reduction': [('mild', 0.7), ('moderate', 0.5), ('severe', 0.3)],
}
SEVERITY_ORDER = {'mild': 1, 'moderate': 2, 'severe': 3}


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


def entropy(probs):
    eps = 1e-12
    return -torch.sum(probs * torch.log2(probs + eps), dim=1)


def denormalize(images):
    mean = CIFAR_MEAN.to(device=images.device, dtype=images.dtype)
    std = CIFAR_STD.to(device=images.device, dtype=images.dtype)
    return images * std + mean


def normalize(images):
    mean = CIFAR_MEAN.to(device=images.device, dtype=images.dtype)
    std = CIFAR_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def apply_corruption(images, corruption_name, parameter):
    pixel_images = torch.clamp(denormalize(images), 0.0, 1.0)

    if corruption_name == 'gaussian_noise':
        corrupted = torch.clamp(pixel_images + torch.randn_like(pixel_images) * parameter, 0.0, 1.0)
    elif corruption_name == 'gaussian_blur':
        kernel_size = [int(parameter), int(parameter)]
        corrupted = TF.gaussian_blur(pixel_images, kernel_size=kernel_size)
    elif corruption_name == 'contrast_reduction':
        mean = pixel_images.mean(dim=(2, 3), keepdim=True)
        corrupted = torch.clamp(mean + parameter * (pixel_images - mean), 0.0, 1.0)
    else:
        raise ValueError(f'Unknown corruption: {corruption_name}')

    return normalize(corrupted)


def evaluate_corruption(model, dataloader, corruption_name, parameter, device):
    criterion = KLDivergenceLoss()
    total_entropy = 0.0
    total_kl = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            corrupted_images = apply_corruption(images, corruption_name, parameter)

            logits = model(corrupted_images)
            probs = F.softmax(logits, dim=-1)
            batch_size = images.size(0)

            total_entropy += entropy(probs).sum().item()
            total_kl += criterion(logits, targets).item() * batch_size
            total_correct += (probs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            total_samples += batch_size

    return {
        'mean_predicted_entropy': total_entropy / total_samples,
        'mean_kl': total_kl / total_samples,
        'top1_accuracy': total_correct / total_samples,
    }


def plot_ood_response(root_dir, results_df):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for corruption_name in CORRUPTION_SPECS:
        subset = results_df[results_df['corruption'] == corruption_name].sort_values('severity_rank')
        ax.plot(
            subset['severity_rank'],
            subset['mean_predicted_entropy'],
            marker='o',
            linewidth=2,
            label=corruption_name.replace('_', ' ').title(),
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Mild', 'Moderate', 'Severe'])
    ax.set_xlabel('Severity')
    ax.set_ylabel('Mean Predicted Entropy')
    ax.set_title('OOD Corruption Response of Predicted Entropy')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = os.path.join(root_dir, 'plots', 'ood_corruption_response.png')
    fig.savefig(output)
    plt.close(fig)
    return output


def run_ood_benchmark():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_dl, _ = get_cifar10h_dataloaders(root_dir, batch_size=128, num_workers=0)
    model = load_best_checkpoint_model(root_dir, device)

    rows = []
    for corruption_name, severity_specs in CORRUPTION_SPECS.items():
        for severity_name, parameter in severity_specs:
            metrics = evaluate_corruption(model, test_dl, corruption_name, parameter, device)
            rows.append(
                {
                    'corruption': corruption_name,
                    'severity': severity_name,
                    'severity_rank': SEVERITY_ORDER[severity_name],
                    'parameter': parameter,
                    **metrics,
                }
            )

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(root_dir, 'ood_robustness_results.csv')
    results_df.to_csv(csv_path, index=False)
    plot_ood_response(root_dir, results_df)

    print(f'Saved OOD robustness results to {csv_path}')


if __name__ == '__main__':
    run_ood_benchmark()