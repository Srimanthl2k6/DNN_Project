import os
import torch
import torch.nn.functional as F
import numpy as np
from data import get_cifar10h_dataloaders
from model import CustomResNet18
from robustness import fgsm_attack, pgd_attack
from losses import KLDivergenceLoss

def entropy(probs):
    eps = 1e-12
    return -torch.sum(probs * torch.log2(probs + eps), dim=1)

def run_benchmark():
    import logging
    import warnings
    warnings.filterwarnings('ignore')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    _, _, test_dl, _ = get_cifar10h_dataloaders(root, batch_size=1, num_workers=0)
    
    # Load indices
    low_idx = set(np.load(os.path.join(root, 'attacks', 'lowest_entropy_indices.npy')))
    high_idx = set(np.load(os.path.join(root, 'attacks', 'highest_entropy_indices.npy')))
    
    # Setup eval model
    model = CustomResNet18(head_type='linear', pretrain_strategy='random').to(device)
    model.eval()
    
    ckpt_dir = os.path.join(root, 'checkpoints')
    if os.path.exists(ckpt_dir):
        pths = [p for p in os.listdir(ckpt_dir) if p.endswith('.pth')]
        if pths:
            model.load_state_dict(torch.load(os.path.join(ckpt_dir, pths[0]), map_location=device, weights_only=True))
    
    criterion = KLDivergenceLoss()
    
    results = {
        'high': {'original': {'acc': 0, 'kl': 0, 'ent': 0}, 'fgsm': {'acc': 0, 'kl': 0, 'ent': 0}, 'pgd': {'acc': 0, 'kl': 0, 'ent': 0}, 'count': 0},
        'low': {'original': {'acc': 0, 'kl': 0, 'ent': 0}, 'fgsm': {'acc': 0, 'kl': 0, 'ent': 0}, 'pgd': {'acc': 0, 'kl': 0, 'ent': 0}, 'count': 0}
    }
    
    epsilon = 0.031
    alpha = 0.01
    iters = 10
    
    for i, (image, target) in enumerate(test_dl):
        if i in high_idx:
            group = 'high'
        elif i in low_idx:
            group = 'low'
        else:
            continue
            
        image, target = image.to(device), target.to(device)
        
        with torch.no_grad():
            logits_orig = model(image)
            prob_orig = F.softmax(logits_orig, dim=-1)
            acc_orig = (prob_orig.argmax(1) == target.argmax(1)).float().mean().item()
            kl_orig = criterion(logits_orig, target).item()
            ent_orig = entropy(prob_orig).mean().item()
            
        results[group]['original']['acc'] += acc_orig
        results[group]['original']['kl'] += kl_orig
        results[group]['original']['ent'] += ent_orig
        results[group]['count'] += 1
        
        # FGSM Attack
        image.requires_grad = True
        logits_for_grad = model(image)
        loss = criterion(logits_for_grad, target)
        model.zero_grad()
        loss.backward()
        data_grad = image.grad.detach()
        adv_fgsm = fgsm_attack(image.detach(), epsilon, data_grad)
        image.requires_grad = False
        
        with torch.no_grad():
            logits_fgsm = model(adv_fgsm)
            prob_fgsm = F.softmax(logits_fgsm, dim=-1)
            results[group]['fgsm']['acc'] += (prob_fgsm.argmax(1) == target.argmax(1)).float().mean().item()
            results[group]['fgsm']['kl'] += criterion(logits_fgsm, target).item()
            results[group]['fgsm']['ent'] += entropy(prob_fgsm).mean().item()
            
        # PGD Attack
        adv_pgd = pgd_attack(model, image, target, epsilon, alpha, iters, criterion)
        with torch.no_grad():
            logits_pgd = model(adv_pgd)
            prob_pgd = F.softmax(logits_pgd, dim=-1)
            results[group]['pgd']['acc'] += (prob_pgd.argmax(1) == target.argmax(1)).float().mean().item()
            results[group]['pgd']['kl'] += criterion(logits_pgd, target).item()
            results[group]['pgd']['ent'] += entropy(prob_pgd).mean().item()

    print("================== ROBUSTNESS BENCHMARK ==================")
    import pandas as pd
    flat_results = []
    
    for group in ['high', 'low']:
        cnt = results[group]['count']
        if cnt == 0: continue
        print(f"\n--- {group.upper()} ENTROPY SAMPLES (N={cnt}) ---")
        orig_acc = results[group]['original']['acc']/cnt
        orig_kl = results[group]['original']['kl']/cnt
        orig_ent = results[group]['original']['ent']/cnt
        print(f"Original -> Acc: {orig_acc:.4f}, KL: {orig_kl:.4f}, Ent: {orig_ent:.4f}")
        
        for atk in ['fgsm', 'pgd']:
            acc = results[group][atk]['acc']/cnt
            kl = results[group][atk]['kl']/cnt
            ent = results[group][atk]['ent']/cnt
            
            acc_drop = orig_acc - acc
            kl_shift = kl - orig_kl
            ent_shift = ent - orig_ent
            
            print(f"{atk.upper()}     -> Acc: {acc:.4f} (Drop: {acc_drop:+.4f}), KL: {kl:.4f} (Shift: {kl_shift:+.4f}), Ent: {ent:.4f} (Shift: {ent_shift:+.4f})")
            
            flat_results.append({
                'entropy_group': group,
                'attack': atk,
                'acc_drop': acc_drop,
                'kl_shift': kl_shift,
                'ent_shift': ent_shift
            })
            
    pd.DataFrame(flat_results).to_csv(os.path.join(root, 'robustness_results.csv'), index=False)

if __name__ == '__main__':
    run_benchmark()