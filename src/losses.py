import torch
import torch.nn as torch_nn
import torch.nn.functional as F

class KLDivergenceLoss(torch_nn.Module):
    """
    Computes the Kullback-Leibler divergence between predicted and true distributions.
    Expects logits as input for predictions, and probabilities for targets.
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        # Note: KLDivLoss in PyTorch expects log_probabilities as inputs, and probabilities as targets
        # reduction='batchmean' divides by batch size.
        self.kl_loss = torch_nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, target_probs):
        log_preds = F.log_softmax(logits, dim=-1)
        return self.kl_loss(log_preds, target_probs)

class JSDivergenceLoss(torch_nn.Module):
    """
    Computes Jensen-Shannon divergence.
    JSD(P||Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)
    """
    def __init__(self):
        super(JSDivergenceLoss, self).__init__()
        self.kl = torch_nn.KLDivLoss(reduction='none')

    def forward(self, logits, target_probs):
        preds = F.softmax(logits, dim=-1)
        
        # M = 0.5 * (P + Q)
        m = 0.5 * (preds + target_probs)
        
        log_m = torch.log(m + 1e-12)
        log_preds = F.log_softmax(logits, dim=-1)
        log_target = torch.log(target_probs + 1e-12)
        
        # Calculate manually to ensure PyTorch's native KLDivLoss reduction doesn't 
        # sever the gradients of the dynamically shifting `preds` target parameter.
        kl_1 = torch.sum(target_probs * (log_target - log_m), dim=-1)
        kl_2 = torch.sum(preds * (log_preds - log_m), dim=-1)
        
        jsd = 0.5 * kl_1 + 0.5 * kl_2
        return jsd.mean()

class SoftCrossEntropyLoss(torch_nn.Module):
    """
    Soft-target cross-entropy.
    CE(P, Q) = - sum(P * log(Q)) where P is target_probs and Q is softmax(logits)
    """
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, logits, target_probs):
        log_preds = F.log_softmax(logits, dim=-1)
        return -torch.sum(target_probs * log_preds, dim=-1).mean()

class CustomDisagreementLoss(torch_nn.Module):
    """
    A custom loss function combining KL divergence and an entropy error penalty.
    Forces the model to predict the overall uncertainty (entropy) correctly.
    Loss = KL(P||Q) + alpha * MSE(Entropy(P), Entropy(Q))
    """
    def __init__(self, alpha=1.0):
        super(CustomDisagreementLoss, self).__init__()
        self.kl_loss = torch_nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha

    def _entropy(self, probs):
        # probs must not be zero
        return -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1)

    def forward(self, logits, target_probs):
        log_preds = F.log_softmax(logits, dim=-1)
        kl = self.kl_loss(log_preds, target_probs)
        
        preds = F.softmax(logits, dim=-1)
        pred_entropy = self._entropy(preds)
        target_entropy = self._entropy(target_probs)
        
        entropy_mse = F.mse_loss(pred_entropy, target_entropy)
        
        return kl + self.alpha * entropy_mse

if __name__ == "__main__":
    logits = torch.randn(4, 10, requires_grad=True)
    targets = F.softmax(torch.randn(4, 10), dim=-1)
    
    kl = KLDivergenceLoss()
    jsd = JSDivergenceLoss()
    ce = SoftCrossEntropyLoss()
    custom = CustomDisagreementLoss(alpha=0.5)
    
    print(f"KL Loss: {kl(logits, targets).item()}")
    print(f"JSD Loss: {jsd(logits, targets).item()}")
    print(f"CE Loss: {ce(logits, targets).item()}")
    print(f"Custom Loss: {custom(logits, targets).item()}")
