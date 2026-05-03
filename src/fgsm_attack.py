import os
import torch
import numpy as np
import pandas as pd
from model import CustomResNet18
from data import get_cifar10h_dataloaders
from robustness import fgsm_attack
from losses import KLDivergenceLoss
from evaluate_auto import entropy

def fgsm_pipeline(model, dataloader, target_indices, epsilons, device, root):
    model.eval()
    
    # Store results per epsilon
    results = {eps: {'kl_drop': 0, 'acc_drop': 0, 'ent_change': 0} for eps in epsilons}
    
    loss_fn = KLDivergenceLoss()
    
    valid_indices = set(target_indices)
    
    for i, (images, targets) in enumerate(dataloader):
        # We process batch size 1 here to manually index
        images, targets = images.to(device), targets.to(device)
        images.requires_grad = True
        
        logits = model(images)
        loss = loss_fn(logits, targets)
        model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        
        # original metrics
        orig_preds = torch.exp(torch.nn.functional.log_softmax(logits, dim=-1))
        orig_acc = (orig_preds.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean().item()
        orig_kl = torch.sum(targets * (torch.log(targets + 1e-12) - torch.nn.functional.log_softmax(logits, dim=-1)), dim=1).mean().item()
        orig_ent = entropy(orig_preds.detach().cpu().numpy()).mean()
        
        for eps in epsilons:
            perturbed_images = fgsm_attack(images, eps, data_grad)
            adv_logits = model(perturbed_images)
            adv_preds = torch.exp(torch.nn.functional.log_softmax(adv_logits, dim=-1))
            
            adv_acc = (adv_preds.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean().item()
            adv_kl = torch.sum(targets * (torch.log(targets + 1e-12) - torch.nn.functional.log_softmax(adv_logits, dim=-1)), dim=1).mean().item()
            adv_ent = entropy(adv_preds.detach().cpu().numpy()).mean()
            
            results[eps]['kl_drop'] += (orig_kl - adv_kl)
            results[eps]['acc_drop'] += (orig_acc - adv_acc)
            results[eps]['ent_change'] += (adv_ent - orig_ent)

    # Note: Just appending as mock data aggregation for now, will calculate full loop later.
    return results

if __name__ == "__main__":
    print("FGSM pipeline loaded. Will attach epsilons = [0.01, 0.03, 0.05, 0.1] to subset batches.")
    # Implement subset FGSM extraction once model weights finish loading
