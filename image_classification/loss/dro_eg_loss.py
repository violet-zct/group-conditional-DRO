import torch
import torch.nn as nn


class DROEGLoss(nn.Module):
    def __init__(self, model, n_groups, n_domains, step_size, split_with_group=True, normalize=False):
        super(DROEGLoss, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.lr = step_size
        self.n_groups = n_groups
        self.n_domains = n_domains
        self.split_with_group = split_with_group
        self.normalize = normalize
        if self.split_with_group:
            self.register_buffer('adv_probs', torch.ones(self.n_groups) / self.n_groups)
        else:
            self.register_buffer('adv_probs', torch.ones(self.n_domains) / self.n_domains)

    def reset(self):
        self.adv_prob.fill_(1.0)
        self.adv_prob.div_(self.adv_prob.sum())

    def forward(self, x, y, g, d=None, w=None):
        outputs = self.model(x)
        # batch size
        losses = self.criterion(outputs, y)
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
        gdro_counts = zero_vec.scatter_add(0, s, one_vec).float()
        gdro_losses = zero_vec.scatter_add(0, s, losses).div(gdro_counts + (gdro_counts == 0).float())

        if self.training:
            adjusted_losses = gdro_losses.detach()
            if self.normalize:
                adjusted_losses = adjusted_losses / (adjusted_losses.sum())
            exp_weights = torch.exp(self.lr * adjusted_losses)
            self.adv_probs.mul_(exp_weights)
            self.adv_probs.div_(self.adv_probs.sum())
        robust_loss = (gdro_losses * self.adv_probs).sum()

        with torch.no_grad():
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

    def compute_loss(self, x, y):
        outputs = self.model(x)
        losses = self.criterion(outputs, y)
        return losses
