import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from fairseq import metrics, utils
from fairseq import utils
from collections import defaultdict
import os
from . import FairseqCriterion, register_criterion

import logging
logger = logging.getLogger(__name__)


def convert_to_list(st, t):
    return list(map(t, st.strip().split(',')))

@register_criterion('cross_entropy_group_dro')
class GroupDROCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.args = args

        self.device = torch.cuda.current_device()

        self.n_train_groups = task.args.num_train_groups
        self.n_test_groups = task.args.num_test_groups

        if args.dynamic_group_dro:
            # run GC-DRO
            self.n_train_groups = args.num_train_groups

        self.vocab_dict = task.target_dictionary.string
        self.step_size = args.eg_step_size
        self.temp_idx = 0
        self.group_counts = None

        self.task_type = "mnli"

        if args.log_internal:
            self.flog = open(os.path.join(args.save_dir, "outer_log.txt"), "w")
        else:
            self.flog = None

        self.inter_updates = 0

    def initialize(self, n_groups):
        self.n_train_groups = n_groups
        logger.info("Group num = {}".format(self.n_train_groups))
        self.register_buffer('eta', torch.zeros(1))
        if self.step_size < 0:
            self.EMA_alpha = self.args.ema
            self.alpha = self.args.dro_alpha
            if self.alpha <= 0.20:
                self.alpha = 0.20
            if self.args.baselines is None:
                self.loss_baselines = torch.Tensor([0. for _ in range(self.n_train_groups)]).to(self.device)
            else:
                self.loss_baselines = torch.Tensor(convert_to_list(self.args.baselines, float)).to(self.device)
            #
            self.register_buffer('h_fun', torch.ones(self.n_train_groups))
            self.register_buffer('sum_losses', torch.zeros(self.n_train_groups))  # historical loss sum over category
            self.register_buffer('count_cat', torch.ones(self.n_train_groups))
            #
        else:
            self.adj = self.args.adj
            self.register_buffer('adv_probs', torch.ones(self.n_train_groups) / self.n_train_groups)
            self.normalize_loss = self.args.eg_normalize

        self.idx_dict = defaultdict(lambda: len(self.idx_dict))  # autoincrementing index.
        for i in range(self.n_train_groups):
            _ = self.idx_dict['['+str(i)+']']

    def add_args(parser):
        # greedy dro
        parser.add_argument('--dro-alpha', default=1., type=float, help='alpha value for the CVar DRO loss.')
        parser.add_argument('--baselines', default=None, type=str, help='baseline loss values.')

        # exponentiated dro
        parser.add_argument('--eg-step-size', type=float, default=-1,
                            help="step size when using full simplex as the uncertainty set")
        parser.add_argument('--eg-normalize', type=int, default=0)
        parser.add_argument('--adj', default=-1, type=float, help='weights for adjusted group DRO')
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')

    def reset_history(self):
        if self.step_size < 0:
            self.h_fun.fill_(1.)
            self.sum_losses.fill_(0.)
        else:
            self.adv_probs.fill_(1.)
        self.inter_updates += 1

    def update_mw(self):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.sum_losses
        baselined_losses = past_losses - self.loss_baselines
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

        self.temp_idx += 1
        if self.temp_idx % 100 == 0:
            logger.info("EMA past losses: {}".format(past_losses[0:self.n_train_groups]))
            # logger.info("Baseline losses: {}".format(baselined_losses[0:self.n_train_groups]))
            logger.info("EMA group fractions: {}".format(past_frac[0:self.n_train_groups]))
            logger.info("Group loss weights: {}".format(self.h_fun[0:self.n_train_groups]))

        if self.temp_idx % 100 == 0 and self.flog is not None:
            fracs = " ".join(["{:.6f}".format(l.item()) for l in past_frac[0:self.n_train_groups]])
            weights = " ".join(["{:.6f}".format(l.item()) for l in self.h_fun[0:self.n_train_groups]])
            self.flog.write("inter_updates={}\tfrac={}\tweights={}\n".format(self.inter_updates, fracs, weights))
            self.flog.flush()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='none',
        )
        ind_loss = loss
        sample_size = targets.numel()
        preds = logits.argmax(dim=1)
        comp = (preds == targets)
        ncorrect = utils.item(comp.sum())
        batch_size = targets.numel()

        if not self.training:
            loss = ind_loss.sum()
            fg_labels = sample['labels']
        else:
            # get labels of examples
            index = sample['labels']
            fg_labels = sample['labels_fg']

            if self.step_size > 0:
                # eg dro
                if "weights" in sample:
                    ind_loss = ind_loss * sample["weights"]
                group_losses, group_counts = self.compute_group_loss(ind_loss, index)
                group_losses, reduce_group_loss = self.compute_group_avg(group_losses, group_counts)
                loss = self.compute_eg_robust_loss(group_losses, reduce_group_loss)
                # final loss is sentence level or token level
                sample_size = 1
            else:
                if "weights" in sample:
                    ind_loss = ind_loss * sample["weights"]

                group_losses, group_counts = self.compute_group_loss(ind_loss, index)
                denom = group_losses.ne(0).sum()
                reduce_group_losses = group_losses.detach().clone()
                if torch.cuda.device_count() > 1:
                    torch.distributed.all_reduce(group_counts)
                    torch.distributed.all_reduce(reduce_group_losses)

                group_denom = group_counts + 1e-8
                reduce_group_losses = reduce_group_losses / group_denom
                # group_losses = group_losses * self.args.distributed_world_size / group_denom / denom

                valid_index = reduce_group_losses.ne(0)
                self.sum_losses[valid_index] = self.sum_losses[valid_index].mul(1 - self.EMA_alpha).add(reduce_group_losses[valid_index], alpha=self.EMA_alpha)
                self.count_cat[valid_index] = self.count_cat[valid_index].mul(1 - 0.05).add(group_counts[valid_index], alpha=0.05)
                self.update_mw()
                sample_size = batch_size
                loss = (ind_loss * self.h_fun[index]).sum()
                #sample_size = 1
                # loss = (group_losses * self.h_fun).sum()

            one_vec = torch.ones(batch_size, device='cuda')  # B
            zero_vec = torch.zeros(self.n_train_groups, device='cuda')
            group_acc = zero_vec.scatter_add(0, index, comp.float())
            group_count = zero_vec.scatter_add(0, index, one_vec)

        fg_one_vec = torch.ones(batch_size, device='cuda')  # B
        fg_zero_vec = torch.zeros(self.n_test_groups, device='cuda')
        fg_group_acc = fg_zero_vec.scatter_add(0, fg_labels, comp.float())
        fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, fg_one_vec)

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': batch_size,
            'sample_size': sample_size,
            'ncorrect': ncorrect,
            'n_train_groups': self.n_train_groups,
            'n_test_groups': self.n_test_groups,
            'gpu_count': 1
        }

        if self.training:
            for ii in range(self.n_train_groups):
                logging_output['w{}'.format(ii)] = self.adv_probs[ii].data if self.step_size > 0 else (self.h_fun[ii].data / self.alpha)
                logging_output['l{}'.format(ii)] = reduce_group_loss[ii].data if self.step_size > 0 else self.sum_losses[ii].data
                logging_output['gcorrect{}'.format(ii)] = group_acc[ii].data
                logging_output['gcount{}'.format(ii)] = group_count[ii].data

        for ii in range(self.n_test_groups):
            logging_output["fg_gcorrect{}".format(ii)] = fg_group_acc[ii].data
            logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data

        return loss, sample_size, logging_output

    def compute_group_loss(self, ind_loss, index):
        zero_vec = torch.zeros(self.n_train_groups, device='cuda')  # G
        group_losses = zero_vec.scatter_add(0, index, ind_loss)

        one_vec = torch.ones(ind_loss.size(0), device='cuda')  # B
        group_counts = zero_vec.scatter_add(0, index, one_vec)
        return group_losses, group_counts

    def compute_group_avg(self, group_losses, group_counts):
        reduce_group_losses = group_losses.detach().clone()
        if torch.cuda.device_count() > 1:
            torch.distributed.all_reduce(group_counts)
            torch.distributed.all_reduce(reduce_group_losses)

        group_denom = group_counts + (group_counts == 0).float()  # avoid nans
        reduce_group_losses = reduce_group_losses / group_denom
        group_losses = group_losses * self.args.distributed_world_size / group_denom
        return group_losses, reduce_group_losses

    def compute_eg_robust_loss(self, group_loss, reduce_group_loss):
        adjusted_loss = reduce_group_loss
        if self.adj > 0 and self.group_counts is not None:
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        exp_weights = torch.exp(self.step_size * adjusted_loss)
        self.adv_probs.mul_(exp_weights)
        self.adv_probs.div_(self.adv_probs.sum())

        self.temp_idx += 1
        if self.temp_idx % 100 == 0 and self.args.distributed_rank == 0:
            logger.info("EG Weights = {}".format(exp_weights / exp_weights.max()))
        robust_loss = group_loss * self.adv_probs
        return robust_loss.sum()

    def compute_train_individual_losses(self, model, sample):
        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        ind_loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='none',
        )
        data_idx = sample['id']
        return ind_loss, data_idx

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))

        if sample_size >= 1:
            metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

        gpu_counts = utils.item(sum(log.get('gpu_count', 0) for log in logging_outputs))
        n_train_groups = sum(log.get('n_train_groups', 0) for log in logging_outputs) / gpu_counts
        n_train_groups = int(n_train_groups.item()) if torch.is_tensor(n_train_groups) else int(n_train_groups)
        n_test_groups = sum(log.get('n_test_groups', 0) for log in logging_outputs) / gpu_counts
        n_test_groups = int(n_test_groups.item()) if torch.is_tensor(n_test_groups) else int(n_test_groups)

        if len(logging_outputs) > 0 and 'w1' in logging_outputs[0]:
            for ii in range(n_train_groups):
                group_loss = sum(log.get('l{}'.format(ii), 0) for log in logging_outputs) / gpu_counts / math.log(2)
                metrics.log_scalar('acl{}'.format(ii), group_loss, 1, round=3)

            for ii in range(n_train_groups):
                group_loss = sum(log.get('l{}'.format(ii), 0) for log in logging_outputs) / gpu_counts / math.log(2)
                metrics.log_scalar('l{}'.format(ii), group_loss, 0, round=3)

            for ii in range(n_train_groups):
                weight = sum(log.get('w{}'.format(ii), 0) for log in logging_outputs) / gpu_counts
                metrics.log_scalar('w{}'.format(ii), weight, 1, round=3)

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
