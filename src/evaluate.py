import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from data import get_cifar10h_dataloaders
from model import CustomResNet18

def entropy(probs):
    eps = 1e-12
    return -np.sum(probs * np.log2(probs + eps), axis=1)

def precision_at_k(true_entropies, pred_entropies, k_values=[100, 200, 500]):
    sorted_true_idx = np.argsort(true_entropies)[::-1]
    sorted_pred_idx = np.argsort(pred_entropies)[::-1]
    results = {}
    for k in k_values:
        top_k_true = set(sorted_true_idx[:k])
        top_k_pred = set(sorted_pred_idx[:k])
        results[f'P@{k}'] = len(top_k_true.intersection(top_k_pred)) / k
    return results

def compute_sba(preds, targets):
    pred_2nd = np.argsort(preds, axis=1)[:, -2]
    target_2nd = np.argsort(targets, axis=1)[:, -2]
    return np.mean(pred_2nd == target_2nd)

def expected_calibration_error(preds, targets, n_bins=15):
    preds_hard = np.max(preds, axis=1)
    targets_hard = np.argmax(targets, axis=1)
    pred_classes = np.argmax(preds, axis=1)
    
    accuracies = (pred_classes == targets_hard)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (preds_hard > bin_boundaries[i]) & (preds_hard <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(preds_hard[in_bin])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(in_bin) / len(preds))
    return ece

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds_log, all_targets, all_preds = [], [], []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            log_preds = F.log_softmax(logits, dim=-1)
            preds = torch.exp(log_preds)
            
            all_preds_log.append(log_preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    preds_log = np.concatenate(all_preds_log, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    # 1. Distribution Matching
    kl_div = np.sum(targets * (np.log(targets + 1e-12) - preds_log), axis=1)
    mean_kl = np.mean(kl_div)
    std_kl = np.std(kl_div)
    
    tvd = 0.5 * np.sum(np.abs(preds - targets), axis=1)
    mean_tvd = np.mean(tvd)

    brier_score = np.mean(np.sum((preds - targets) ** 2, axis=1))
    mean_cos = np.mean(
        np.sum(targets * preds, axis=1) /
        (np.linalg.norm(targets, axis=1) * np.linalg.norm(preds, axis=1) + 1e-12)
    )
    
    # 2. Hard Accuracy Evaluators
    hard_targets = np.argmax(targets, axis=1)
    hard_preds = np.argmax(preds, axis=1)
    top1_acc = accuracy_score(hard_targets, hard_preds)

    # 3. Entropy Prediction Quality
    true_ent = entropy(targets)
    pred_ent = entropy(preds)
    try:
        pearson_corr, _ = pearsonr(true_ent, pred_ent)
        spearman_corr, _ = spearmanr(true_ent, pred_ent)
    except Exception:
        pearson_corr, spearman_corr = 0.0, 0.0
    pre_k = precision_at_k(true_ent, pred_ent, [100, 200, 500])
    
    # 4. Uncertainty & Rankings
    sba = compute_sba(preds, targets)
    ece = expected_calibration_error(preds, targets)

    print(f"--- Complete Evaluation Metrics ---")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"KL Divergence: {mean_kl:.4f} ± {std_kl:.4f}")
    print(f"TVD: {mean_tvd:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Cosine Sim: {mean_cos:.4f}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"Pearson r (Entropy): {pearson_corr:.4f}")
    print(f"Spearman rho (Entropy): {spearman_corr:.4f}")
    print(f"Precision@K: {pre_k}")
    print(f"SBA (2nd Best Accuracy): {sba:.4f}")

    return {
        'top1_acc': top1_acc,
        'kl': mean_kl,
        'tvd': mean_tvd,
        'brier': brier_score,
        'cos_sim': mean_cos,
        'ece': ece,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'p_at_k': pre_k,
        'sba': sba
    }

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, _, test_dl, _ = get_cifar10h_dataloaders(root, batch_size=128, num_workers=2)

    checkpoints_dir = os.path.join(root, 'checkpoints')
    
    experiments = [
        ('Exp_KL_Random_Linear', 'linear', 'random'),
        ('Exp_JS_Random_Linear', 'linear', 'random'),
        ('Exp_SoftCE_Random_Linear', 'linear', 'random'),
        ('Exp_CustomDisag_Random_Linear', 'linear', 'random'),
        ('Exp_KL_Random_MLP', 'mlp', 'random'),
        ('Exp_KL_ImageNet_Linear', 'linear', 'imagenet'),
    ]
    
    for exp_name, head_type, pretrain_strategy in experiments:
        ckpt_path = os.path.join(checkpoints_dir, f'{exp_name}.pth')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {exp_name}")
            continue
            
        print(f"\nEvaluating {exp_name}...")
        try:
            model = CustomResNet18(head_type=head_type, pretrain_strategy=pretrain_strategy)
        except Exception as exc:
            if pretrain_strategy == 'imagenet':
                raise RuntimeError(
                    f"Failed to load ImageNet pretrained weights while evaluating {exp_name}. "
                    "Check internet access or the local torchvision weights cache."
                ) from exc
            raise
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.to(device)
        
        evaluate_model(model, test_dl, device)
