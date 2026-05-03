import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    """
    Fast Gradient Sign Method (FGSM) attack.
    """
    sign_data_grad = data_grad.sign()
    
    device = image.device
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
    
    epsilon_scaled = epsilon / std
    perturbed_image = image + epsilon_scaled * sign_data_grad
    
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    
    perturbed_image = torch.max(torch.min(perturbed_image, max_val), min_val)
    return perturbed_image

def pgd_attack(model, images, labels, epsilon, alpha, iters, loss_fn):
    """
    Projected Gradient Descent (PGD) attack.
    """
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True
    
    for _ in range(iters):
        outputs = model(perturbed_images)
        loss = loss_fn(outputs, labels)
        
        model.zero_grad()
        if perturbed_images.grad is not None:
            perturbed_images.grad.zero_()
        loss.backward()
        
        data_grad = perturbed_images.grad.detach()
        
        device = perturbed_images.device
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        
        alpha_scaled = alpha / std
        epsilon_scaled = epsilon / std
        
        perturbed_images = perturbed_images.detach() + alpha_scaled * data_grad.sign()
        
        min_val = (0.0 - mean) / std
        max_val = (1.0 - mean) / std
        
        eta = torch.clamp(perturbed_images - original_images, min=-epsilon_scaled, max=epsilon_scaled)
        perturbed_images = torch.max(torch.min(original_images + eta, max_val), min_val)
        perturbed_images.requires_grad = True
        
    return perturbed_images
