# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

        self.n_train_groups = task.args.num_train_groups
        self.n_test_groups = task.args.num_test_groups
        self.device = torch.cuda.current_device()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def loss(self, model, net_output, sample):
        reduce = False if self.training and "labels" in sample else True
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        reduce_group_loss = None
        if not reduce:
            mask = (sample['target'] != self.padding_idx).float()
            token_losses = loss.reshape_as(sample['target'])
            ind_loss = (token_losses * mask).sum(1)
            ind_loss = ind_loss / mask.sum(1)
            index = sample['labels']
            _, reduce_group_loss = self.compute_group_avg(ind_loss, index.unsqueeze(0))

            loss = loss.sum()
            nll_loss = nll_loss.sum()
        return loss, nll_loss, reduce_group_loss

    def compute_group_avg(self, losses, group_idx):
        # unmodified loss is sentence level loss, the group loss returned is used to compute the robust loss,
        # the normalized_group_loss is returned to update self.adv_prob
        # unmodified_losses, losses might be the same
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_train_groups).unsqueeze(1).long().to(self.device)).float()  # G X B
        group_count = group_map.sum(1)
        group_loss = group_map @ losses.view(-1)  # G
        reduce_group_loss = group_loss.detach()

        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = group_loss / group_denom  # G / #G

        if torch.cuda.device_count() > 1:
            torch.distributed.all_reduce(group_count)
            torch.distributed.all_reduce(reduce_group_loss)

        reduce_group_denom = group_count + (group_count == 0).float()  # avoid nans
        reduce_group_loss = reduce_group_loss / reduce_group_denom
        return group_loss, reduce_group_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, reduce_group_loss = self.loss(model, net_output, sample)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'n_train_groups': self.n_train_groups,
            'gpu_count': 1
        }
        if reduce_group_loss is not None:
            for ii in range(self.n_train_groups):
                logging_output['nll_{}'.format(ii)] = reduce_group_loss[ii].data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        gpu_counts = utils.item(sum(log.get('gpu_count', 0) for log in logging_outputs))
        ngroups = sum(log.get('n_train_groups', 0) for log in logging_outputs) / gpu_counts
        ngroups = int(ngroups.item()) if torch.is_tensor(ngroups) else int(ngroups)

        if len(logging_outputs) > 0 and "nll_0" in logging_outputs[0]:
            gpu_counts = utils.item(sum(log.get('gpu_count', 0) for log in logging_outputs))
            for ii in range(ngroups):
                group_loss_token = sum(log.get('nll_{}'.format(ii), 0) for log in logging_outputs) / gpu_counts
                metrics.log_scalar('gnll_{}'.format(ii), group_loss_token / math.log(2), 1, round=3)
                metrics.log_scalar('gppl_{}'.format(ii), 2 ** (group_loss_token / math.log(2)), 0, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
