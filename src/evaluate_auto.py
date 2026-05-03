import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
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
        if i == 0:
            in_bin = (preds_hard >= bin_boundaries[i]) & (preds_hard <= bin_boundaries[i+1])
        else:
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
    
    tvd = 0.5 * np.sum(np.abs(preds - targets), axis=1)
    mean_tvd = np.mean(tvd)

    brier_score = np.mean(np.sum((preds - targets) ** 2, axis=1))
    
    # Vectorized cosine similarity avoiding list comprehension
    num = np.sum(targets * preds, axis=1)
    den = np.linalg.norm(targets, axis=1) * np.linalg.norm(preds, axis=1)
    mean_cos = np.mean(num / np.clip(den, 1e-12, None))
    
    # 2. Hard Accuracy Evaluators
    hard_targets = np.argmax(targets, axis=1)
    hard_preds = np.argmax(preds, axis=1)
    top1_acc = accuracy_score(hard_targets, hard_preds)

    # 3. Entropy Prediction Quality
    true_ent = entropy(targets)
    pred_ent = entropy(preds)
    pearson_corr, _ = pearsonr(true_ent, pred_ent) if np.std(pred_ent) > 0 else (0.0, 0.0)
    spearman_corr, _ = spearmanr(true_ent, pred_ent) if np.std(pred_ent) > 0 else (0.0, 0.0)
    pre_k = precision_at_k(true_ent, pred_ent, [100, 200, 500])
    
    # 4. Uncertainty & Rankings
    sba = compute_sba(preds, targets)
    ece = expected_calibration_error(preds, targets)

    return {
        'kl': mean_kl,
        'tvd': mean_tvd,
        'brier': brier_score,
        'cosine': mean_cos,
        'acc': top1_acc,
        'sba': sba,
        'ece': ece,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'p@100': pre_k['P@100'],
        'p@200': pre_k['P@200'],
        'p@500': pre_k['P@500']
    }

def poll_checkpoints(root_dir, device, test_dl):
    results_path = os.path.join(root_dir, 'results.csv')
    checkpoints_dir = os.path.join(root_dir, 'checkpoints')
    
    evaluated_models = set()
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        if 'experiment_name' in df.columns:
            evaluated_models = set(df['experiment_name'].tolist())
    else:
        # Create CSV with headers
        cols = ['experiment_name', 'loss', 'head', 'kl', 'tvd', 'brier', 'cosine', 'acc', 'sba', 'ece', 'pearson', 'spearman', 'p@100', 'p@200', 'p@500']
        pd.DataFrame(columns=cols).to_csv(results_path, index=False)
            
    # Quick scan of what's inside
    if not os.path.exists(checkpoints_dir):
        return
        
    for pth_file in os.listdir(checkpoints_dir):
        if not pth_file.endswith('.pth'):
            continue
            
        exp_name = pth_file.replace('.pth', '')
        
        if exp_name in evaluated_models:
            continue
            
        print(f"\n[AUTO-EVAL] Evaluating new checkpoint: {exp_name}...")
        
        # We need to infer architecture. For this project, mostly linear / random.
        # But we could parse from exp_name.
        head_type = 'linear'
        pretrain = 'random'
        loss_type = 'Unknown'
        if 'ImageNet' in exp_name: pretrain = 'imagenet'
        if 'KL' in exp_name: loss_type = 'KL'
        if 'CustomDisag' in exp_name: loss_type = 'CustomDisag'
        
        model = CustomResNet18(head_type=head_type, pretrain_strategy=pretrain)
        ckpt_path = os.path.join(checkpoints_dir, pth_file)
        
        # Load weights safely
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.to(device)
            metrics = evaluate_model(model, test_dl, device)
            
            # Format row
            row = {
                'experiment_name': exp_name,
                'loss': loss_type,
                'head': head_type,
                **metrics
            }
            pd.DataFrame([row]).to_csv(results_path, mode='a', header=False, index=False)
            print(f"--> Saved evaluation metrics for {exp_name} to results.csv")
            evaluated_models.add(exp_name)
        except Exception as e:
            print(f"Failed to evaluate {exp_name}: {str(e)}")

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")
    
    _, _, test_dl, _ = get_cifar10h_dataloaders(root, batch_size=128, num_workers=0)
    
    # Run once to process existing
    poll_checkpoints(root, device, test_dl)
