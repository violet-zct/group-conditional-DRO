import torch
import torch.nn as nn


class ERMLoss(nn.Module):
    def __init__(self, model, n_groups):
        super(ERMLoss, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.n_groups = n_groups

    def forward(self, x, y, g, d=None, w=None):
        outputs = self.model(x)
        losses = self.criterion(outputs, y)
        # compute loss for each group.
        with torch.no_grad():
            batch_size = losses.size(0)
            zero_vec = losses.new_zeros(self.n_groups)
            one_vec = losses.new_ones(batch_size)
            group_counts = zero_vec.scatter_add(0, g, one_vec).float()

            group_losses = zero_vec.scatter_add(0, g, losses)
            group_losses.div_(group_counts + (group_counts == 0).float())

            preds = outputs.argmax(dim=1)
            corrects = preds.eq(y).float()
            acc = corrects.sum().mul_(100. / batch_size)
            group_accs = zero_vec.scatter_add(0, g, corrects).mul_(100.).div_(group_counts + (group_counts == 0).float())

        return losses.mean(), acc, group_losses, group_accs, group_counts

    def compute_loss(self, x, y):
        outputs = self.model(x)
        losses = self.criterion(outputs, y)
        return losses
