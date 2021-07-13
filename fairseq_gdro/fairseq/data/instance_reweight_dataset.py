# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from collections import OrderedDict
from . import BaseWrapperDataset, plasma_utils
from . import LanguagePairDataset, NestedDictionaryDataset
import torch
import os
from collections import Counter
import logging
logger = logging.getLogger(__name__)


class InstanceReweightDataset(BaseWrapperDataset):
    """Split a given dataset into two groups: a uniformly reweighed group and an ERM group.

    Args:
        dataset (~torch.utils.data.AddLabelDataset): dataset on which to sample.
        uniform_size: the equal size in the reweighed group.
        batch_by_size (bool): whether or not to batch by sequence length
            (default: True).
        seed (int): RNG seed to use (default: 0).
        epoch (int): starting epoch number (default: 0).
    """

    def __init__(
        self,
        args,
        dataset,
        label_dataset,
        resplit_greedy=False,
        batch_by_size=True,
        seed=0,
        distributed_rank=-1,
        label_dataset_fg=None,
    ):
        super().__init__(dataset)
        self.args = args
        self.distributed_rank = distributed_rank

        self.beta = args.beta_cover_instances
        self.ema = args.ema if args.beta_ema < 0 else args.beta_ema

        self.label_dataset = label_dataset
        self.internal_batch_by_size = batch_by_size
        self.seed = seed

        self._cur_epoch = None
        self._cur_resplit_epoch = None
        self._resplit_resampling_weights = None

        self.original_groups_indices = self.original_split()

        self.resplit_greedy = resplit_greedy
        self.num_groups = len(self.original_groups_indices)
        self.num_splits = self.num_groups

        self.split_counts = np.array([len(self.original_groups_indices[i]) for i in range(self.num_groups)])

        self.resplit(epoch=0)
        self.label_dataset_fg = label_dataset_fg
        self.accum_losses = None

        if args.log_internal:
            self.flog = open(os.path.join(args.save_dir, "inter_log.txt"), "w")
            self.update_num = 0
        else:
            self.flog = None

    def __getitem__(self, index):
        convert_index = self._cur_indices.array[index]
        item = self.dataset[convert_index]
        item['label'] = self.label_dataset[convert_index]
        item['weight'] = self.weight_array[convert_index]
        if self.label_dataset_fg is not None:
            item['label_fg'] = self.label_dataset_fg[convert_index]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return None
        sample = self.dataset.collater(samples)
        if "labels" not in sample:
            # in the case that the base dataset doesn't handle label collation, e.g. nested dataset
            # labels are the actual training labels, which might be noisy
            sample["labels"] = torch.LongTensor([s['label'] for s in samples])
            if self.label_dataset_fg is not None:
                # fg labels means fine-grained labels that are created from A x Y, A is the spurious attributes
                sample["labels_fg"] = torch.LongTensor([s['label_fg'] for s in samples])
        if "weights" not in sample:
            sample["weights"] = torch.FloatTensor([s['weight'] for s in samples])
        return sample

    @property
    def sizes(self):
        if isinstance(self.dataset.sizes, list):
            return [s[self._cur_indices.array] for s in self.dataset.sizes]
        return self.dataset.sizes[self._cur_indices.array]

    def num_tokens(self, index):
        return self.dataset.num_tokens(self._cur_indices.array[index])

    def size(self, index):
        return self.dataset.size(self._cur_indices.array[index])

    def ordered_indices(self):
        if self.internal_batch_by_size:
            if isinstance(self.sizes, np.ndarray) and len(self.sizes.shape) > 1:
                order = [
                    np.arange(len(self)),
                    self.sizes[:, 1],
                    self.sizes[:, 0]
                ]
            else:
                order = [
                    np.arange(len(self)),
                    self.sizes[0],
                ]  # No need to handle `self.shuffle == True`
            return np.lexsort(order)
        else:
            return np.arange(len(self))

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.dataset.prefetch(self._cur_indices.array[indices])

    def original_split(self):
        data_groups = OrderedDict()
        for idx in range(len(self.dataset)):
            label = self.label_dataset[idx]
            if label not in data_groups:
                data_groups[label] = [idx]
            else:
                data_groups[label].append(idx)
        logger.info("Groups: {}".format(data_groups.keys()))
        return data_groups

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self._cur_epoch = epoch

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def resplit(self, epoch, losses=None):
        # reweight dataset
        logger.info(f"---- Re-Weight at Epoch = {epoch} -----")
        self._cur_resplit_epoch = epoch
        rng = np.random.RandomState(
            [
                42,  # magic number
                self.seed % (2 ** 32),  # global seed
                self._cur_resplit_epoch,  # epoch index
            ]
        )
        split_array = np.array(self.label_dataset.labels)

        if losses is not None:
            if self.accum_losses is None:
                self.accum_losses = losses
            else:
                self.accum_losses = self.accum_losses * (1 - self.ema) + losses * self.ema

            #
            total = len(split_array)
            if self.flog is not None:
                self.update_num += 1
                self.flog.write("Update={}\n".format(self.update_num))

            for gidx in range(self.num_groups):
                # 0 1 4 7 9
                select_idx = np.where(split_array == gidx)[0]
                count = len(select_idx)
                idx_sorted = np.argsort(self.accum_losses[select_idx])
                idx = select_idx[idx_sorted][::-1]
                cutoff_count = int((total - count) * count * self.beta / (total - count * self.beta))
                self.weight_array[idx] = count / total
                self.weight_array[idx[:cutoff_count]] = 1.0 / self.beta
            #
                if self.flog is not None:
                    losses = " ".join(["{:.6f}".format(l) for l in self.accum_losses[idx]])
                    weight = " ".join(["{:.6f}".format(l) for l in self.weight_array[idx]])
                    labels = " ".join(["{}".format(self.dataset[l]["target"].item()) for l in idx])
                    self.flog.write("group_id={}\tcount={}\tlosses={}\tweights={}\tlabels={}\n".format(gidx, len(select_idx), losses, weight, labels))
        else:
            self.weight_array = np.ones(len(self.label_dataset.labels))

        def _compute_weights(labels):
            label_counts = Counter(labels)
            self.split_data_sizes = {}
            for key in label_counts.keys():
                self.split_data_sizes[key] = label_counts[key]
            logger.info("Split group size: {}".format(self.split_data_sizes))
            weights = [len(labels) * 1.0 / label_counts[ll] for ll in labels]
            weights_arr = np.array(weights, dtype=np.float64)
            weights_arr /= weights_arr.sum()
            weights = plasma_utils.PlasmaArray(weights_arr)
            return weights

        if self.resplit_greedy:
            self._cur_indices = plasma_utils.PlasmaArray(np.arange(len(self)))
        else:
            if self._resplit_resampling_weights is None:
                self._resplit_resampling_weights = _compute_weights(split_array)

            self._cur_indices = plasma_utils.PlasmaArray(
                rng.choice(
                    len(self.dataset),
                    len(self.dataset),
                    replace=True,
                    p=self._resplit_resampling_weights.array,
                )
            )