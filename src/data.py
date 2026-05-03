import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def load_cifar10_test_batch(data_dir):
    """Loads the CIFAR-10 test_batch file containing the 10,000 test images."""
    file_path = os.path.join(data_dir, 'test_batch')
    with open(file_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    images = dict[b'data']
    labels = dict[b'labels']
    # CIFAR-10 images are 10000x3072, reshape to 10000x3x32x32
    images = images.reshape(10000, 3, 32, 32)
    # Transpose to 10000x32x32x3 for easier PIL/ToTensor operations (wait, PyTorch ToTensor expects HWC if numpy, but maybe we can just work directly with tensors)
    images = images.transpose(0, 2, 3, 1) # HWC
    return images, np.array(labels)

class CIFAR10HDataset(Dataset):
    def __init__(self, images, probs, transform=None):
        """
        images: numpy array of shape (N, 32, 32, 3) (HWC format)
        probs: numpy array of shape (N, 10), representing soft labels (human distributions)
        transform: torchvision transforms
        """
        self.images = images
        self.probs = probs
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_arr = self.images[idx]
        prob = self.probs[idx]
        
        # We need to pass uint8 numpy arrays or PIL Images to transforms usually
        if self.transform is not None:
            # Need to ensure img_arr is uint8 if ToTensor or PIL transforms are used
            img_arr = img_arr.astype(np.uint8)
            img = self.transform(img_arr)
        else:
            img = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0

        target_prob = torch.from_numpy(prob).float()
        return img, target_prob

def get_cifar10h_dataloaders(root_dir, batch_size=64, num_workers=2, seed=42):
    """
    Loads images, aligns with soft labels, splits into 6000 train, 2000 val, 2000 test.
    """
    # Paths
    test_batch_dir = os.path.join(root_dir, 'cifar-10-python', 'cifar-10-batches-py')
    probs_path = os.path.join(root_dir, 'cifar10h-probs.npy')
    
    # Load data
    images, _ = load_cifar10_test_batch(test_batch_dir)
    probs = np.load(probs_path)
    
    # Assertions for sanity checks
    assert images.shape == (10000, 32, 32, 3), f"Images shape mismatch: {images.shape}"
    assert probs.shape == (10000, 10), f"Probs shape mismatch: {probs.shape}"
    
    # Ensure probs sum to 1
    prob_sums = np.sum(probs, axis=1)
    assert np.allclose(prob_sums, 1.0), "Probs do not sum to 1.0"
    
    # Generate splits
    indices = np.arange(10000)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_idx = indices[:6000]
    val_idx = indices[6000:8000]
    test_idx = indices[8000:]
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    full_dataset_train = CIFAR10HDataset(images, probs, transform=train_transform)
    full_dataset_eval = CIFAR10HDataset(images, probs, transform=eval_transform)
    
    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_eval, val_idx)
    test_dataset = Subset(full_dataset_eval, test_idx)
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader, probs

def compute_entropy(probs):
    """Computes Shannon Entropy for a batch of probability distributions."""
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    return -np.sum(probs * np.log2(probs + eps), axis=1)

def eda_visualizations(root_dir, probs):
    """Generates the required Phase 1 EDA visualizations."""
    entropies = compute_entropy(probs)
    
    os.makedirs(os.path.join(root_dir, 'plots'), exist_ok=True)
    
    # 1. Histogram of entropy
    plt.figure(figsize=(8, 5))
    plt.hist(entropies, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Human Annotator Entropy (Disagreement)')
    plt.xlabel('Shannon Entropy (max ~3.32)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(root_dir, 'plots', 'entropy_histogram.png'))
    plt.close()
    
    print(f"Saved entropy_histogram.png. Mean entropy: {np.mean(entropies):.4f}")

if __name__ == '__main__':
    # Test data script
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Loading data from {root}...")
    train_dl, val_dl, test_dl, all_probs = get_cifar10h_dataloaders(root, batch_size=32, num_workers=0)
    print(f"Train subset: {len(train_dl.dataset)} samples")
    print(f"Val subset: {len(val_dl.dataset)} samples")
    print(f"Test subset: {len(test_dl.dataset)} samples")
    
    for x, y in train_dl:
        print(f"Batch X shape: {x.shape}, Batch Y shape: {y.shape}")
        break

    eda_visualizations(root, all_probs)
    print("EDA Complete. Shape sanity checks passed.")
