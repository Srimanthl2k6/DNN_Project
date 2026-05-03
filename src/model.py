import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import torchvision.models as models

class CustomResNet18(torch_nn.Module):
    def __init__(self, num_classes=10, head_type='linear', pretrain_strategy='random', init_weights=None):
        """
        Adapts ResNet-18 for 32x32 CIFAR-10 images.
        
        args:
            head_type: 'linear' or 'mlp'
            pretrain_strategy: 'random', 'imagenet', or 'cifar10_hard'
            init_weights: path to weights if 'cifar10_hard' is used.
        """
        super(CustomResNet18, self).__init__()
        
        if pretrain_strategy == 'imagenet':
            # load imagenet pretrained
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet18(weights=None)
            
        # Extract features (remove the final fully connected layer)
        self.feature_dim = backbone.fc.in_features
        
        # Modify the first layer for 32x32 image inputs
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.conv1 = torch_nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        # Remove the max pool completely since images are already small
        # We replace it with Identity, so spatial dims are 32x32 entering layer1
        backbone.maxpool = torch_nn.Identity()
        
        # Store backbone modules
        self.backbone = torch_nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, # Identity
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool 
            # Output is (B, 512, 1, 1)
        )
        
        if pretrain_strategy == 'cifar10_hard' and init_weights is not None:
             # Assuming weights are passed and saved after pre-training
             checkpoint = torch.load(init_weights, map_location='cpu')
             # Need to filter out just backbone weights, or handle appropriately
             # For now, placeholder
             self.load_state_dict(checkpoint, strict=False)

        # Build prediction head
        if head_type == 'linear':
            self.head = torch_nn.Linear(self.feature_dim, num_classes)
        elif head_type == 'mlp':
            self.head = torch_nn.Sequential(
                torch_nn.Linear(self.feature_dim, 256),
                torch_nn.ReLU(),
                torch_nn.Dropout(p=0.5),
                torch_nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1) # Flatten (B, 512)
        logits = self.head(features)
        # Goal is predicting probability distribution -> return softmax
        # Note: Depending on loss function design (e.g. KLDivLoss expects log_probabilities)
        # It's cleaner to return raw logits and compute log_softmax in the training step or loss function.
        # Return logits to easily use F.log_softmax in the loss.
        return logits

def print_model_summary():
    model_linear = CustomResNet18(head_type='linear')
    num_params = sum(p.numel() for p in model_linear.parameters() if p.requires_grad)
    print(f"Custom ResNet18 (Linear Head) Trainable Parameters: {num_params}")

    model_mlp = CustomResNet18(head_type='mlp')
    num_params_mlp = sum(p.numel() for p in model_mlp.parameters() if p.requires_grad)
    print(f"Custom ResNet18 (MLP Head) Trainable Parameters: {num_params_mlp}")

if __name__ == "__main__":
    print_model_summary()
    x = torch.randn(2, 3, 32, 32)
    m = CustomResNet18()
    out = m(x)
    print(f"Forward pass output shape (logits): {out.shape}")
