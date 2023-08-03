import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLoss(nn.Module):
    def __init__(self, num_classes, beta=1.0):
        super(AdaLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta

    def forward(self, predictions, targets):
        # Calculate the cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets)

        targets = targets.long()

        # Calculate the AdaLoss
        class_probs = F.softmax(predictions, dim=1)
        pt = class_probs[torch.arange(class_probs.size(0)), targets]
        ada_loss = -torch.log(pt.pow(self.beta))

        # Calculate the final AdaLoss with class balancing
        class_weights = (1.0 - pt).pow(self.beta)
        ada_loss = ada_loss * class_weights
        ada_loss = ada_loss.mean()

        # Combine the AdaLoss and cross-entropy loss
        total_loss = ce_loss + ada_loss
        return total_loss
