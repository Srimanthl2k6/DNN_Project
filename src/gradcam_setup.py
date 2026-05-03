import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import get_cifar10h_dataloaders
from model import CustomResNet18
from explain import GradCAM, overlay_cam_on_image

def run_gradcam(root_dir, device):
    _, _, test_dl, _ = get_cifar10h_dataloaders(root_dir, batch_size=1, num_workers=0)
    
    # Needs a checkpoint to process
    ckpt = os.path.join(root_dir, 'checkpoints', 'Exp_KL_Random_Linear.pth')
    if not os.path.exists(ckpt):
        print("Waiting for baseline checkpoint to run Grad-CAM.")
        return
        
    model = CustomResNet18(head_type='linear', pretrain_strategy='random')
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    target_layer = model.resnet.layer4[-1].conv2
    cam = GradCAM(model, target_layer)
    
    # Load High/Low Entropy index arrays
    attacks_dir = os.path.join(root_dir, 'attacks')
    low_idx_file = os.path.join(attacks_dir, 'lowest_entropy_indices.npy')
    high_idx_file = os.path.join(attacks_dir, 'highest_entropy_indices.npy')
    
    if not os.path.exists(low_idx_file) or not os.path.exists(high_idx_file):
        print("Run find_samples.py first to generate entropy indices.")
        return
        
    low_ent_idx = np.load(low_idx_file)[:5] # just do visual for 5
    high_ent_idx = np.load(high_idx_file)[:5]
    
    target_indices = set(low_ent_idx) | set(high_ent_idx)
    os.makedirs(os.path.join(root_dir, 'plots', 'gradcam'), exist_ok=True)
    
    saved = 0
    for i, (image, target) in enumerate(test_dl):
        if i not in target_indices: continue
        
        image = image.to(device)
        mask = cam(image)
        
        # Display image handling
        disp_img = image.cpu().squeeze().permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        disp_img = std * disp_img + mean
        disp_img = np.clip(disp_img, 0, 1)
        
        result = overlay_cam_on_image(disp_img, mask)
        
        output_prefix = 'high_entropy' if i in high_ent_idx else 'low_entropy'
        plt.imsave(os.path.join(root_dir, f'plots/gradcam/{output_prefix}_{i}.png'), result)
        saved += 1
        if saved >= len(target_indices): break
        
    print(f"Saved {saved} Grad-CAM heatmaps for highest and lowest entropy targets.")
    
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_gradcam(root, device)
