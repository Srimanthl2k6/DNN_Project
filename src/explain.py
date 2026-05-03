import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM for visualizing network activations.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        b, c, h, w = x.size()
        
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        # Weighted combination of activations
        activations = self.activations[0]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=x.device)
        
        for i, w_i in enumerate(weights):
            cam += w_i * activations[i]
            
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (w, h))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam

def overlay_cam_on_image(img, mask):
    """
    Helper function to overlay Grad-CAM heatmap on the original image.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # img is expected to be [0,1] floating point RGB
    cam_result = heatmap + np.float32(img)
    cam_result = cam_result / np.max(cam_result)
    return np.uint8(255 * cam_result)
