import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import CustomResNet18
from data import get_cifar10h_dataloaders
from robustness import fgsm_attack
from losses import KLDivergenceLoss


def entropy(probs):
    eps = 1e-12
    return -torch.sum(probs * torch.log2(probs + eps), dim=1)


def _empty_accumulator():
    return {
        'n_samples': 0,
        'orig_acc': 0.0,
        'adv_acc': 0.0,
        'orig_kl': 0.0,
        'adv_kl': 0.0,
        'orig_entropy': 0.0,
        'adv_entropy': 0.0,
    }


def _metrics(logits, targets, loss_fn):
    probs = F.softmax(logits, dim=-1)
    return {
        'acc': (probs.argmax(dim=-1) == targets.argmax(dim=-1)).float(),
        'kl': torch.sum(
            targets * (torch.log(targets + 1e-12) - F.log_softmax(logits, dim=-1)),
            dim=1,
        ),
        'entropy': entropy(probs),
    }


def run_fgsm_epsilon_sweep(model, dataloader, target_groups, epsilons, device):
    model.eval()
    loss_fn = KLDivergenceLoss()

    index_to_groups = {}
    for group_name, indices in target_groups.items():
        for idx in indices:
            index_to_groups.setdefault(int(idx), []).append(group_name)

    accumulators = {
        (group_name, float(eps)): _empty_accumulator()
        for group_name in target_groups
        for eps in epsilons
    }

    relative_idx = 0
    for images, targets in dataloader:
        batch_size = images.size(0)

        for batch_offset in range(batch_size):
            groups = index_to_groups.get(relative_idx, [])
            if not groups:
                relative_idx += 1
                continue

            image = images[batch_offset:batch_offset + 1].to(device).clone().detach()
            target = targets[batch_offset:batch_offset + 1].to(device)
            image.requires_grad = True

            logits = model(image)
            loss = loss_fn(logits, target)
            model.zero_grad()
            loss.backward()
            data_grad = image.grad.detach()

            with torch.no_grad():
                orig = _metrics(logits.detach(), target, loss_fn)

            for eps in epsilons:
                eps = float(eps)
                perturbed_image = fgsm_attack(image.detach(), eps, data_grad)
                with torch.no_grad():
                    adv_logits = model(perturbed_image)
                    adv = _metrics(adv_logits, target, loss_fn)

                for group_name in groups:
                    acc = accumulators[(group_name, eps)]
                    acc['n_samples'] += 1
                    acc['orig_acc'] += orig['acc'].item()
                    acc['adv_acc'] += adv['acc'].item()
                    acc['orig_kl'] += orig['kl'].item()
                    acc['adv_kl'] += adv['kl'].item()
                    acc['orig_entropy'] += orig['entropy'].item()
                    acc['adv_entropy'] += adv['entropy'].item()

            relative_idx += 1

    rows = []
    for group_name in target_groups:
        for eps in epsilons:
            eps = float(eps)
            acc = accumulators[(group_name, eps)]
            count = acc['n_samples']
            if count == 0:
                continue

            orig_acc = acc['orig_acc'] / count
            adv_acc = acc['adv_acc'] / count
            orig_kl = acc['orig_kl'] / count
            adv_kl = acc['adv_kl'] / count
            orig_entropy = acc['orig_entropy'] / count
            adv_entropy = acc['adv_entropy'] / count

            rows.append({
                'entropy_group': group_name,
                'epsilon': eps,
                'n_samples': count,
                'orig_acc': orig_acc,
                'adv_acc': adv_acc,
                'acc_drop': orig_acc - adv_acc,
                'orig_kl': orig_kl,
                'adv_kl': adv_kl,
                'kl_shift': adv_kl - orig_kl,
                'orig_entropy': orig_entropy,
                'adv_entropy': adv_entropy,
                'entropy_change': adv_entropy - orig_entropy,
            })

    return pd.DataFrame(
        rows,
        columns=[
            'entropy_group',
            'epsilon',
            'n_samples',
            'orig_acc',
            'adv_acc',
            'acc_drop',
            'orig_kl',
            'adv_kl',
            'kl_shift',
            'orig_entropy',
            'adv_entropy',
            'entropy_change',
        ],
    )


def fgsm_pipeline(model, dataloader, target_indices, epsilons, device, root=None):
    if isinstance(target_indices, dict):
        return run_fgsm_epsilon_sweep(model, dataloader, target_indices, epsilons, device)
    return run_fgsm_epsilon_sweep(model, dataloader, {'subset': target_indices}, epsilons, device)

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilons = [0.01, 0.03, 0.05, 0.1]

    ckpt_path = os.path.join(root, 'checkpoints', 'Exp_KL_Random_Linear.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint required for FGSM sweep: {ckpt_path}")

    _, _, test_dl, _ = get_cifar10h_dataloaders(root, batch_size=1, num_workers=0)

    low_idx = np.load(os.path.join(root, 'attacks', 'lowest_entropy_indices.npy'))
    high_idx = np.load(os.path.join(root, 'attacks', 'highest_entropy_indices.npy'))

    model = CustomResNet18(head_type='linear', pretrain_strategy='random')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)

    df = fgsm_pipeline(
        model,
        test_dl,
        {'low': low_idx, 'high': high_idx},
        epsilons,
        device,
        root,
    )

    output_path = os.path.join(root, 'fgsm_epsilon_sweep.csv')
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved FGSM epsilon sweep to {output_path}")
