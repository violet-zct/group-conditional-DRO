from collections import defaultdict
import torch
import torch.nn as nn


class DROGreedyLoss(nn.Module):
    def __init__(self, model, n_groups, n_domains, alpha, fraction=None, split_with_group=True):
        super(DROGreedyLoss, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.ema = 0.1
        self.n_groups = n_groups
        self.n_domains = n_domains
        self.split_with_group = split_with_group
        if self.split_with_group:
            self.n_splits = n_groups
        else:
            self.n_splits = n_domains

        self.register_buffer('h_fun', torch.ones(self.n_splits))
        self.register_buffer('sum_losses', torch.zeros(self.n_splits))  # historical loss sum over category
        if fraction is not None:
            self.register_buffer('fraction', torch.from_numpy(fraction).float())
            self.register_buffer('count_cat', None)
        else:
            self.register_buffer('count_cat', torch.ones(self.n_splits))

        self.idx_dict = defaultdict(lambda: len(self.idx_dict))  # autoincrementing index.
        for i in range(self.n_groups):
            _ = self.idx_dict['[' + str(i) + ']']

    def reset(self):
        self.h_fun.fill_(1.)
        self.sum_losses.fill_(0.)
        if self.count_cat is not None:
            self.count_cat.fill_(1.)

    def reset_loss(self):
        self.h_fun.fill_(1.)
        self.sum_losses.fill_(0.)

    def forward(self, x, y, g, d, w):
        outputs = self.model(x)
        # batch size
        losses = self.criterion(outputs, y) * w
        # compute loss for each group.
        batch_size = losses.size(0)
        one_vec = losses.new_ones(batch_size)
        if self.split_with_group:
            zero_vec = losses.new_zeros(self.n_groups)
            s = g
        else:
            zero_vec = losses.new_zeros(self.n_domains)
            s = d

        # n_groups
        gdro_losses = zero_vec.scatter_add(0, s, losses)
        robust_loss = (gdro_losses * self.h_fun).sum() / batch_size

        with torch.no_grad():
            if self.training:
                gdro_counts = zero_vec.scatter_add(0, s, one_vec).float()
                gdro_losses = gdro_losses.detach().div(gdro_counts + (gdro_counts == 0).float())
                valid_idx = gdro_counts.gt(0)
                self.sum_losses[valid_idx] = self.sum_losses[valid_idx].mul(1 - self.ema).add(gdro_losses[valid_idx], alpha=self.ema)
                if self.count_cat is not None:
                    self.count_cat.mul_(1 - self.ema).add_(gdro_counts, alpha=self.ema)
                self.update_mw()

            zero_vec = losses.new_zeros(self.n_groups)
            one_vec = losses.new_ones(batch_size)
            group_counts = zero_vec.scatter_add(0, g, one_vec).float()
            group_losses = zero_vec.scatter_add(0, g, losses)
            group_losses.div_(group_counts + (group_counts == 0).float())

            preds = outputs.argmax(dim=1)
            corrects = preds.eq(y).float()
            acc = corrects.sum().mul(100. / batch_size)
            group_accs = zero_vec.scatter_add(0, g, corrects).mul(100.).div(group_counts + (group_counts == 0).float())

        return robust_loss, acc, group_losses, group_accs, group_counts

    def update_mw(self):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.sum_losses
        if self.count_cat is not None:
            past_frac = self.count_cat / self.count_cat.sum()
        else:
            past_frac = self.fraction
        sorted_losses, sort_id = torch.sort(past_losses, descending=True)
        sorted_frac = past_frac[sort_id]
        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        self.h_fun = self.h_fun.new_full(self.h_fun.size(), 0.1)
        self.h_fun[sort_id[:cutoff_count]] = 1.0 / self.alpha
        leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
        tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
        self.h_fun[sort_id[cutoff_count]] = max(tiebreak_fraction, 0.1)

    def compute_loss(self, x, y):
        outputs = self.model(x)
        losses = self.criterion(outputs, y)
        return losses
