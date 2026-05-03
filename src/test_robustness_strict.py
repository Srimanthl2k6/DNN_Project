import torch
import torch.nn.functional as F
from robustness import fgsm_attack

# Dummy test with 1 image, 3 channels
device = 'cpu'
image = torch.zeros(1, 3, 32, 32)
data_grad = torch.ones_like(image)
epsilon = 0.031
# Run attack
try:
    adv = fgsm_attack(image, epsilon, data_grad)
    print(f'Epsilon scaling check: {epsilon}')
except Exception as e:
    print(e)
