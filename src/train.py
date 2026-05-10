import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import copy
import json

from data import get_cifar10h_dataloaders
from model import CustomResNet18
from losses import KLDivergenceLoss, JSDivergenceLoss, SoftCrossEntropyLoss, CustomDisagreementLoss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            
    return total_loss / len(dataloader.dataset)

def train(model, train_dl, val_dl, criterion, optimizer, num_epochs=50, patience=10, device='cuda'):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss = evaluate(model, val_dl, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"--> Saved new best model with val loss {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

def run_experiment(exp_name, model_fn, criterion, train_dl, val_dl, device, root_dir):
    print(f"=== Starting Experiment: {exp_name} ===")
    try:
        model = model_fn().to(device)
    except Exception as exc:
        if exp_name == 'Exp_KL_ImageNet_Linear':
            raise RuntimeError(
                "Failed to load ImageNet pretrained weights for "
                "Exp_KL_ImageNet_Linear. Check internet access or the local "
                "torchvision weights cache before rerunning training."
            ) from exc
        raise

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Using a learning rate scheduler could also be added here
    
    best_model, train_losses, val_losses = train(
        model, train_dl, val_dl, criterion, optimizer, num_epochs=10, patience=3, device=device
    )
    
    # Save best model
    os.makedirs(os.path.join(root_dir, 'checkpoints'), exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(root_dir, 'checkpoints', f'{exp_name}.pth'))

    losses_path = os.path.join(root_dir, 'checkpoints', f'{exp_name}_losses.json')
    with open(losses_path, 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'experiment_name': exp_name,
                'train_losses': [float(loss) for loss in train_losses],
                'val_losses': [float(loss) for loss in val_losses],
            },
            handle,
            indent=2,
        )
    
    return train_losses, val_losses


def build_ablations():
    return [
        # (Experiment Name, Head Type, Pretrain Strategy, Loss Criterion)
        ('Exp_KL_Random_Linear', 'linear', 'random', KLDivergenceLoss()),
        ('Exp_JS_Random_Linear', 'linear', 'random', JSDivergenceLoss()),
        ('Exp_SoftCE_Random_Linear', 'linear', 'random', SoftCrossEntropyLoss()),
        ('Exp_CustomDisag_Random_Linear', 'linear', 'random', CustomDisagreementLoss(alpha=0.5)),
        ('Exp_KL_Random_MLP', 'mlp', 'random', KLDivergenceLoss()),
        ('Exp_KL_ImageNet_Linear', 'linear', 'imagenet', KLDivergenceLoss()),
    ]


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Note: To avoid running out of memory quickly or taking forever, adjust batch size
    train_dl, val_dl, test_dl, _ = get_cifar10h_dataloaders(root, batch_size=128, num_workers=2)
    
    # --- ABLATIONS BLOCK ---
    ablations = build_ablations()
    
    for exp_name, head_type, pretrain_strategy, criterion in ablations:
        print(f"\n[{exp_name}] -> Head: {head_type} | Weights: {pretrain_strategy} | Loss: {criterion.__class__.__name__}")
        model_fn = lambda h=head_type, p=pretrain_strategy: CustomResNet18(head_type=h, pretrain_strategy=p)
        run_experiment(exp_name, model_fn, criterion, train_dl, val_dl, device, root)
