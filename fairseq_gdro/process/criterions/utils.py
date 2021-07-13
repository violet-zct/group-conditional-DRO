import torch
import torch.nn.functional as F
import numpy as np


def compute_binary_acc(x, y):
    # x: logit
    y_hat = (x > 0).float()
    acc = (y == y_hat).float().mean().item() * 100
    return acc


def compute_binary_group_acc(x, y, group_id, num_groups, device):
    batch_size = x.size(0)
    preds = (x > 0).float()
    comp = (preds == y)
    fg_one_vec = torch.ones(batch_size, device=device)  # B
    fg_zero_vec = torch.zeros(num_groups, device=device)
    fg_group_acc = fg_zero_vec.scatter_add(0, group_id, comp.float())
    fg_group_count = fg_zero_vec.scatter_add(0, group_id, fg_one_vec)
    group_acc = (fg_group_acc / fg_group_count).cpu().numpy()

    avg_acc = sum(comp).item() / len(comp)
    return (group_acc, avg_acc)