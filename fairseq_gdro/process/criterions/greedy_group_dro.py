import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from process.criterions.utils import compute_binary_group_acc


class GreedyGroupDRO(nn.Module):
    def __init__(self, alpha=0.2, train_groups=2, test_groups=2, device='cuda'):
        super(GreedyGroupDRO, self).__init__()

        self.n_train_groups = train_groups
        self.n_test_groups = test_groups

        self.EMA_alpha = 0.1
        self.alpha = alpha
        self.device = device
        #
        self.register_buffer('h_fun', torch.ones(self.n_train_groups))
        self.register_buffer('sum_losses', torch.zeros(self.n_train_groups))  # historical loss sum over category
        self.register_buffer('count_cat', torch.ones(self.n_train_groups))

    def forward(self, x, y, group_ids, clean_ids=None, weights=None):
        if clean_ids is None:
            clean_ids = group_ids
        ind_loss = bce(x, y, reduction='none')  # B
        acc = compute_binary_group_acc(x, y, clean_ids, self.n_test_groups, self.device)

        if not self.training:
            return acc

        if weights is not None:
            ind_loss = ind_loss * weights
        group_loss, group_counts = self.compute_group_loss(ind_loss, group_ids)
        reduce_group_losses = group_loss / (group_counts + 1e-8)
        valid_index = reduce_group_losses.ne(0)
        self.sum_losses[valid_index] = self.sum_losses[valid_index].mul(1 - self.EMA_alpha).add(
            reduce_group_losses[valid_index], alpha=self.EMA_alpha)
        self.count_cat[valid_index] = self.count_cat[valid_index].mul(1 - 0.05).add(group_counts[valid_index],
                                                                                    alpha=0.05)
        self.update_mw()
        loss = (ind_loss * self.h_fun[group_ids]).mean()
        return loss, acc

    def update_mw(self):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        baselined_losses = self.sum_losses
        past_frac = self.count_cat / self.count_cat.sum()  # p_train_t
        #
        sorted_losses, sort_id = torch.sort(baselined_losses, descending=True)
        sorted_frac = past_frac[sort_id]
        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        self.h_fun = self.h_fun.new_full(self.h_fun.size(), 0.1)
        self.h_fun[sort_id[:cutoff_count]] = 1.0 / self.alpha
        leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
        tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
        self.h_fun[sort_id[cutoff_count]] = tiebreak_fraction
        # print("---- frac = {} -----".format(past_frac.cpu().numpy()))
        # print("---- weights = {} ----".format(self.h_fun.cpu().numpy()))

    def compute_group_loss(self, ind_loss, index):
        zero_vec = torch.zeros(self.n_train_groups, device=self.device)  # G
        group_losses = zero_vec.scatter_add(0, index, ind_loss)

        one_vec = torch.ones(ind_loss.size(0), device=self.device)  # B
        group_counts = zero_vec.scatter_add(0, index, one_vec)
        return group_losses, group_counts