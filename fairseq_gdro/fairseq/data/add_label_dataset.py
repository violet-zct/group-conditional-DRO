from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np

from . import FairseqDataset
from . import BaseWrapperDataset
from . import RawLabelDataset
import torch


class AddLabelDataset(BaseWrapperDataset):
    def __init__(self,
                 label_dataset:RawLabelDataset,
                 dataset:FairseqDataset,
                 label_2:RawLabelDataset=None):

        super().__init__(dataset)

        self.dataset = dataset
        self.label_dataset = label_dataset
        self.label_dataset_2 = label_2

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['label'] = self.label_dataset[idx]
        if self.label_dataset_2 is not None:
            # Hack for MNLI: for train, label2 is the finegrained label, label is the original resplit label
            # for test/valid, label2 is none, and label is finegrained label
            item['label_fg'] = self.label_dataset_2[idx]
        return item

    def collater(self, samples):
        sample = self.dataset.collater(samples)
        if len(sample) == 0:
            return sample
        if "labels" not in sample:
            # in the case that the base dataset doesn't handle label collation, e.g. nested dataset
            sample["labels"] = torch.LongTensor([s['label'] for s in samples])
            if self.label_dataset_2 is not None:
                sample["labels_fg"] = torch.LongTensor([s['label_fg'] for s in samples])
        return sample