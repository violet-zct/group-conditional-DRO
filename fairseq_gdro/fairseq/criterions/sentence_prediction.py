# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

        self.n_train_groups = task.args.num_train_groups
        self.n_test_groups = task.args.num_test_groups

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()
        preds = logits.argmax(dim=1)
        comp = preds == targets

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction='sum')

        # # todo: ad-hoc change
        if not self.training:
            fg_labels = sample['labels']
        else:
            fg_labels = sample['labels_fg']  # test groups
            index = sample['labels']  # train groups
            one_vec = torch.ones(sample_size, device='cuda')  # B
            zero_vec = torch.zeros(self.n_test_groups, device='cuda')
            group_acc = zero_vec.scatter_add(0, index, comp.float())
            group_count = zero_vec.scatter_add(0, index, one_vec)

        fg_one_vec = torch.ones(sample_size, device='cuda')  # B
        fg_zero_vec = torch.zeros(self.n_test_groups, device='cuda')
        fg_group_acc = fg_zero_vec.scatter_add(0, fg_labels, comp.float())
        fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, fg_one_vec)

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
            'n_train_groups': self.n_train_groups,
            'n_test_groups': self.n_test_groups,
            'gpu_count': 1
        }

        if not self.regression_target:
            logging_output['ncorrect'] = comp.sum()

        if self.training:
            for ii in range(self.n_train_groups):
                logging_output['gcorrect{}'.format(ii)] = group_acc[ii].data
                logging_output['gcount{}'.format(ii)] = group_count[ii].data
        else:
            for ii in range(self.n_test_groups):
                logging_output["fg_gcorrect{}".format(ii)] = fg_group_acc[ii].data
                logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

        gpu_counts = utils.item(sum(log.get('gpu_count', 0) for log in logging_outputs))
        n_train_groups = sum(log.get('n_train_groups', 0) for log in logging_outputs) / gpu_counts
        n_train_groups = int(n_train_groups.item()) if torch.is_tensor(n_train_groups) else int(n_train_groups)
        n_test_groups = sum(log.get('n_test_groups', 0) for log in logging_outputs) / gpu_counts
        n_test_groups = int(n_test_groups.item()) if torch.is_tensor(n_test_groups) else int(n_test_groups)

        if len(logging_outputs) > 0 and 'gcorrect0' in logging_outputs[0]:
            for ii in range(n_train_groups):
                g_ncorrect = sum(log.get('gcorrect{}'.format(ii), 0) for log in logging_outputs)
                g_nsents = utils.item(sum(log.get('gcount{}'.format(ii), 0) for log in logging_outputs))
                division_g_nsents = g_nsents if g_nsents > 0 else 1
                metrics.log_scalar('gacc{}'.format(ii), 100.0 * g_ncorrect / division_g_nsents, g_nsents, round=1)

        if len(logging_outputs) > 0 and 'fg_gcorrect0' in logging_outputs[0]:
            for ii in range(n_test_groups):
                g_ncorrect = sum(log.get('fg_gcorrect{}'.format(ii), 0) for log in logging_outputs)
                g_nsents = utils.item(sum(log.get('fg_gcount{}'.format(ii), 0) for log in logging_outputs))
                division_g_nsents = g_nsents if g_nsents > 0 else 1
                metrics.log_scalar('fg_gacc{}'.format(ii), 100.0 * g_ncorrect / division_g_nsents, g_nsents, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
