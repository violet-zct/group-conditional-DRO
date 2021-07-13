import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from process.criterions.utils import compute_binary_group_acc


class ERM_BCE(nn.Module):
    def __init__(self, device, n_groups):
        super(ERM_BCE, self).__init__()
        self.n_groups = n_groups
        self.device = device

    def forward(self, x, y, group_ids):
        """
        x: model output
        y: target
        """
        acc = compute_binary_group_acc(x, y, group_ids, self.n_groups, self.device)
        if not self.training:
            return acc
        loss = bce(x, y, reduction='mean')
        return loss, acc