import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import CIFAR10_CLASSES, get_cifar10h_dataloaders
from evaluate_auto import entropy
from model import CustomResNet18, save_model_assets


CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616])
BEST_CHECKPOINT = 'Exp_KL_Random_Linear.pth'


def denormalize_cifar_image(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * CIFAR_STD + CIFAR_MEAN
    return np.clip(image, 0.0, 1.0)


def load_checkpoint_state_dict(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_best_checkpoint_model(root_dir, device):
    checkpoint_path = os.path.join(root_dir, 'checkpoints', BEST_CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")

    model = CustomResNet18(head_type='linear', pretrain_strategy='random').to(device)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))
    model.eval()
    return model


def collect_samples_by_relative_index(dataloader, indices):
    requested = [int(idx) for idx in indices]
    pending = set(requested)
    found = {}
    cursor = 0

    for images, targets in dataloader:
        batch_size = images.size(0)
        for batch_offset in range(batch_size):
            relative_idx = cursor + batch_offset
            if relative_idx in pending:
                found[relative_idx] = (images[batch_offset], targets[batch_offset])
        cursor += batch_size

        if pending.issubset(found.keys()):
            break

    missing = [idx for idx in requested if idx not in found]
    if missing:
        raise IndexError(f"Could not find test dataloader-relative indices: {missing}")

    return [found[idx] for idx in requested]


def collect_images_by_relative_index(dataloader, indices):
    return [image for image, _ in collect_samples_by_relative_index(dataloader, indices)]


def run_inference(model, dataloader, device):
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for images, batch_targets in dataloader:
            logits = model(images.to(device))
            predictions.append(F.softmax(logits, dim=-1).cpu().numpy())
            targets.append(batch_targets.cpu().numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def plot_entropy_grid(root_dir, test_dl):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    low_idx = np.load(os.path.join(root_dir, 'attacks', 'lowest_entropy_indices.npy'))
    high_idx = np.load(os.path.join(root_dir, 'attacks', 'highest_entropy_indices.npy'))

    low_display_idx = low_idx[:5]
    high_display_idx = high_idx[-5:][::-1]
    low_samples = collect_samples_by_relative_index(test_dl, low_display_idx)
    high_samples = collect_samples_by_relative_index(test_dl, high_display_idx)

    fig, axes = plt.subplots(
        4,
        5,
        figsize=(17, 10),
        gridspec_kw={'height_ratios': [3.0, 1.6, 3.0, 1.6]},
    )
    fig.suptitle("Low-Entropy Examples (Top) and High-Entropy Examples (Bottom)")

    for title, row_samples, row_indices, image_axes, bar_axes in [
        ('Low entropy', low_samples, low_display_idx, axes[0], axes[1]),
        ('High entropy', high_samples, high_display_idx, axes[2], axes[3]),
    ]:
        for col, ((image, target_probs), idx, image_ax, bar_ax) in enumerate(
            zip(row_samples, row_indices, image_axes, bar_axes)
        ):
            image_ax.imshow(denormalize_cifar_image(image))
            image_ax.set_title(
                f"{title} #{col + 1}\nidx {int(idx)} | H={float(entropy(target_probs.unsqueeze(0).numpy())[0]):.2f}",
                fontsize=10,
            )
            image_ax.axis('off')

            bar_ax.bar(np.arange(len(CIFAR10_CLASSES)), target_probs.numpy(), color='steelblue')
            bar_ax.set_ylim(0.0, 1.0)
            bar_ax.set_xticks(np.arange(len(CIFAR10_CLASSES)))
            bar_ax.set_xticklabels(range(len(CIFAR10_CLASSES)), fontsize=6)
            if col == 0:
                bar_ax.set_ylabel('Prob.', fontsize=8)
            else:
                bar_ax.set_yticklabels([])
            bar_ax.tick_params(axis='y', labelsize=7)

    fig.text(
        0.5,
        0.015,
        'Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck',
        ha='center',
        fontsize=9,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    output = os.path.join(root_dir, 'plots', 'entropy_examples_grid.png')
    legacy_output = os.path.join(root_dir, 'plots', 'entropy_grid.png')
    fig.savefig(output)
    fig.savefig(legacy_output)
    plt.close(fig)
    return output


def plot_entropy_scatter(root_dir, test_dl, device):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    model = load_best_checkpoint_model(root_dir, device)
    predicted_probs, true_probs = run_inference(model, test_dl, device)

    true_entropies = entropy(true_probs)
    predicted_entropies = entropy(predicted_probs)
    lower = float(min(true_entropies.min(), predicted_entropies.min()))
    upper = float(max(true_entropies.max(), predicted_entropies.max()))

    plt.figure(figsize=(7, 7))
    plt.scatter(true_entropies, predicted_entropies, alpha=0.35, s=18, color='teal', edgecolors='none')
    plt.plot([lower, upper], [lower, upper], linestyle='--', color='black', linewidth=1.0)
    plt.title('Predicted vs. True Entropy on CIFAR-10H Test Set')
    plt.xlabel('True Shannon Entropy')
    plt.ylabel('Predicted Shannon Entropy')
    plt.tight_layout()

    output = os.path.join(root_dir, 'plots', 'entropy_scatter.png')
    plt.savefig(output)
    plt.close()
    return output


def plot_loss_curves(root_dir):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    checkpoints_dir = os.path.join(root_dir, 'checkpoints')
    history_files = []
    checkpoint_names = []

    if os.path.isdir(checkpoints_dir):
        history_files = sorted(
            file_name for file_name in os.listdir(checkpoints_dir) if file_name.endswith('_losses.json')
        )
        checkpoint_names = sorted(
            file_name[:-4] for file_name in os.listdir(checkpoints_dir) if file_name.endswith('.pth')
        )

    fig, ax = plt.subplots(figsize=(12, 7))

    available_histories = set()
    if history_files:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(history_files), 1)))
        for color, file_name in zip(colors, history_files):
            file_path = os.path.join(checkpoints_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)

            exp_name = payload.get('experiment_name', file_name.replace('_losses.json', ''))
            train_losses = payload.get('train_losses', [])
            val_losses = payload.get('val_losses', [])
            available_histories.add(exp_name)

            if train_losses:
                ax.plot(
                    np.arange(1, len(train_losses) + 1),
                    train_losses,
                    label=f'{exp_name} train',
                    color=color,
                    linewidth=2,
                )
            if val_losses:
                ax.plot(
                    np.arange(1, len(val_losses) + 1),
                    val_losses,
                    label=f'{exp_name} val',
                    color=color,
                    linestyle='--',
                    linewidth=2,
                )

        ax.set_title('Training and Validation Loss Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.axis('off')
        ax.text(
            0.5,
            0.55,
            'Loss history was not captured for pre-existing runs.',
            ha='center',
            va='center',
            fontsize=13,
        )

    missing_histories = sorted(set(checkpoint_names) - available_histories)
    if missing_histories:
        note = 'Loss history was not captured for pre-existing runs: ' + ', '.join(missing_histories)
        ax.text(
            0.01,
            0.01,
            note,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=9,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'lightgray'},
        )

    fig.tight_layout()
    output = os.path.join(root_dir, 'plots', 'loss_curves.png')
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_loss_comparison(root_dir):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    results_path = os.path.join(root_dir, 'results_final.csv')
    output = os.path.join(root_dir, 'plots', 'loss_comparison.png')

    if not os.path.exists(results_path):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.text(0.5, 0.5, 'results_final.csv not found. Run the evaluation summary pipeline first.', ha='center', va='center')
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        return output

    df = pd.read_csv(results_path)
    cosine_column = 'cosine' if 'cosine' in df.columns else 'cos_sim' if 'cos_sim' in df.columns else None
    required_columns = {'experiment_name', 'kl', 'ece'}
    if cosine_column is None or not required_columns.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.text(0.5, 0.5, 'results_final.csv is missing one of: experiment_name, kl, ece, cosine.', ha='center', va='center')
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        return output

    x_positions = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_positions - width, df['kl'], width=width, label='KL', color='#4c78a8')
    ax.bar(x_positions, df['ece'], width=width, label='ECE', color='#f58518')
    ax.bar(x_positions + width, df[cosine_column], width=width, label='Cosine Similarity', color='#54a24b')
    ax.set_title('KL, ECE, and Cosine Similarity by Experiment')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Metric Value')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['experiment_name'], rotation=35, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_ablation_kl_summary(root_dir):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    results_path = os.path.join(root_dir, 'results_final.csv')
    output = os.path.join(root_dir, 'plots', 'ablation_kl_summary.png')

    if not os.path.exists(results_path):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.text(0.5, 0.5, 'results_final.csv not found. Run the evaluation summary pipeline first.', ha='center', va='center')
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        return output

    df = pd.read_csv(results_path)
    if 'experiment_name' not in df.columns or 'kl' not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.text(0.5, 0.5, 'results_final.csv is missing experiment_name or kl.', ha='center', va='center')
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        return output

    plot_df = df[['experiment_name', 'kl']].dropna().sort_values('kl').reset_index(drop=True)
    colors = ['#2a9d8f'] + ['#9ecae1'] * max(len(plot_df) - 1, 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(plot_df['experiment_name'], plot_df['kl'], color=colors, edgecolor='black')
    ax.set_title('Ablation Summary: KL Divergence Across Experiments')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df['experiment_name'], rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars, plot_df['kl']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_robustness_drops(root_dir):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    res_path = os.path.join(root_dir, 'robustness_results.csv')
    df = pd.read_csv(res_path)
    required = {'entropy_group', 'attack', 'acc_drop'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"robustness_results.csv missing columns: {sorted(missing)}")

    pivot = df.pivot(index='entropy_group', columns='attack', values='acc_drop')
    pivot = pivot.reindex(index=[idx for idx in ['low', 'high'] if idx in pivot.index])
    pivot.plot(kind='bar', figsize=(8, 5))
    plt.title('Robustness Accuracy Drop')
    plt.xlabel('Entropy Group')
    plt.ylabel('Accuracy Drop')
    plt.xticks(rotation=0)
    plt.tight_layout()

    output = os.path.join(root_dir, 'plots', 'robustness_drops.png')
    plt.savefig(output)
    plt.close()
    return output


def generate_visualizations(root_dir):
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_model_assets(root_dir)
    
    # Load dataset
    _, _, test_dl, all_probs = get_cifar10h_dataloaders(root_dir, batch_size=128, num_workers=0)
    
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
    plot_entropy_grid(root_dir, test_dl)
    plot_entropy_scatter(root_dir, test_dl, device)
    plot_loss_curves(root_dir)
    plot_loss_comparison(root_dir)
    plot_ablation_kl_summary(root_dir)

    # 4. Robustness plots
    res_path = os.path.join(root_dir, 'robustness_results.csv')
    if os.path.exists(res_path):
        plot_robustness_drops(root_dir)

    print("Visualizations generated in plots/")

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_visualizations(root)
