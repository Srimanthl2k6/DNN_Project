import csv
import os

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

def _count_trainable_params(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def get_model_summary_rows():
    rows = []
    for model_variant, head_type in [
        ('Custom ResNet18 (Linear Head)', 'linear'),
        ('Custom ResNet18 (MLP Head)', 'mlp'),
    ]:
        model = CustomResNet18(head_type=head_type, pretrain_strategy='random')
        rows.append(
            {
                'model_variant': model_variant,
                'head_type': head_type,
                'pretrain_strategy': 'random',
                'trainable_params': _count_trainable_params(model),
                'backbone_params': _count_trainable_params(model.backbone),
                'head_params': _count_trainable_params(model.head),
            }
        )
    return rows


def save_model_summary_assets(root_dir, summary_rows):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    csv_path = os.path.join(root_dir, 'model_parameter_summary.csv')
    image_path = os.path.join(root_dir, 'plots', 'model_parameter_summary.png')
    fieldnames = [
        'model_variant',
        'head_type',
        'pretrain_strategy',
        'trainable_params',
        'backbone_params',
        'head_params',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, ax = plt.subplots(figsize=(11, 2.5 + 0.55 * len(summary_rows)))
    ax.axis('off')
    table = ax.table(
        cellText=[
            [
                row['model_variant'],
                row['head_type'],
                row['pretrain_strategy'],
                f"{row['trainable_params']:,}",
                f"{row['backbone_params']:,}",
                f"{row['head_params']:,}",
            ]
            for row in summary_rows
        ],
        colLabels=[
            'Model Variant',
            'Head Type',
            'Pretrain',
            'Trainable Params',
            'Backbone Params',
            'Head Params',
        ],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax.set_title('CustomResNet18 Parameter Count Summary', fontsize=14, pad=16)
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return csv_path, image_path


def plot_model_architecture(root_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    output_path = os.path.join(root_dir, 'plots', 'model_architecture.png')

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def draw_box(x, y, width, height, text, facecolor):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle='round,pad=0.012,rounding_size=0.02',
            linewidth=1.4,
            edgecolor='#1f2937',
            facecolor=facecolor,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=10)

    def draw_arrow(x_start, y_start, x_end, y_end):
        arrow = FancyArrowPatch(
            (x_start, y_start),
            (x_end, y_end),
            arrowstyle='-|>',
            mutation_scale=12,
            linewidth=1.4,
            color='#374151',
        )
        ax.add_patch(arrow)

    top_boxes = [
        (0.03, 0.67, 'Input Image\n3 x 32 x 32', '#dbeafe'),
        (0.16, 0.67, 'Stem\nConv3x3 + BN + ReLU', '#bfdbfe'),
        (0.29, 0.67, 'Layer1\n64 channels', '#c7d2fe'),
        (0.42, 0.67, 'Layer2\n128 channels', '#c7d2fe'),
        (0.55, 0.67, 'Layer3\n256 channels', '#c7d2fe'),
        (0.68, 0.67, 'Layer4\n512 channels', '#c7d2fe'),
        (0.81, 0.67, 'AvgPool\nFlatten to 512', '#ddd6fe'),
    ]

    box_width = 0.1
    box_height = 0.14
    for x_pos, y_pos, label, color in top_boxes:
        draw_box(x_pos, y_pos, box_width, box_height, label, color)

    for current_box, next_box in zip(top_boxes, top_boxes[1:]):
        draw_arrow(current_box[0] + box_width, current_box[1] + box_height / 2, next_box[0], next_box[1] + box_height / 2)

    draw_box(0.73, 0.36, 0.16, 0.14, 'Linear Head\nFC 512 -> 10', '#fde68a')
    draw_box(0.92, 0.36, 0.07, 0.14, '10 logits', '#fca5a5')
    draw_box(0.73, 0.10, 0.16, 0.14, 'MLP Head\nFC 512 -> 256\nReLU + Dropout\nFC 256 -> 10', '#f9a8d4')
    draw_box(0.92, 0.10, 0.07, 0.14, '10 logits', '#fca5a5')

    draw_arrow(0.86, 0.67, 0.81, 0.50)
    draw_arrow(0.86, 0.67, 0.81, 0.24)
    draw_arrow(0.89, 0.43, 0.92, 0.43)
    draw_arrow(0.89, 0.17, 0.92, 0.17)

    ax.text(0.5, 0.93, 'CustomResNet18 for CIFAR-10H', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(
        0.5,
        0.88,
        'ResNet-18 backbone adapted for 32x32 inputs: 3x3 stem convolution, no max-pooling, shared backbone with two head options.',
        ha='center',
        va='center',
        fontsize=10,
    )
    ax.text(0.5, 0.58, 'Shared backbone', ha='center', va='center', fontsize=11, fontweight='bold', color='#4338ca')
    ax.text(0.86, 0.55, 'Alternative prediction heads', ha='center', va='center', fontsize=11, fontweight='bold', color='#92400e')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


def print_model_summary(root_dir=None):
    summary_rows = get_model_summary_rows()
    for row in summary_rows:
        print(
            f"{row['model_variant']} Trainable Parameters: {row['trainable_params']:,} "
            f"(backbone={row['backbone_params']:,}, head={row['head_params']:,})"
        )

    if root_dir is not None:
        csv_path, image_path = save_model_summary_assets(root_dir, summary_rows)
        print(f"Saved parameter summary CSV to {csv_path}")
        print(f"Saved parameter summary image to {image_path}")

    return summary_rows


def save_model_assets(root_dir):
    summary_rows = print_model_summary(root_dir=root_dir)
    architecture_path = plot_model_architecture(root_dir)
    print(f"Saved architecture diagram to {architecture_path}")
    return summary_rows, architecture_path

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_model_assets(root)
    x = torch.randn(2, 3, 32, 32)
    m = CustomResNet18()
    out = m(x)
    print(f"Forward pass output shape (logits): {out.shape}")
