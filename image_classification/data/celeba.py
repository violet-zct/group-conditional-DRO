import os
import json
import pandas as pd
import numpy as np

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, root_dir, split, transform=None, init=False):
        # Read in attributes
        self.root_dir = root_dir
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        split_df = pd.read_csv(os.path.join(root_dir, 'list_eval_partition.csv'))
        split_array = split_df['partition'].values == self.split_dict[split]

        attrs_df = pd.read_csv(os.path.join(root_dir, 'list_attr_celeba.csv'))
        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root_dir, 'img_align_celeba')
        self.filename_array = attrs_df['image_id'].values[split_array]
        self.confounder_array = attrs_df['Male'].values[split_array]
        self.y_array = attrs_df['Blond_Hair'].values[split_array]

        self.y_array[self.y_array == -1] = 0
        self.confounder_array[self.confounder_array == -1] = 0
        self.confounder_array = 1 - self.confounder_array

        self.y_dict = {0: "Dark", 1: "Blond"}
        self.confounder_dict = {0: "Male", 1: "Female"}

        self.group_names = ['Male-Dark', 'Female-Dark', 'Male-Blond', 'Female-Blond']
        self.group_dict = {'Male-Dark': 0, 'Female-Dark': 1,
                           'Male-Blond': 2, 'Female-Blond': 3}
        self.group_array = np.array([self.group_dict[self.get_groupname(c, y)] for c, y in zip(self.confounder_array, self.y_array)])
        if init:
            self.domain_array = self.domain_split(split)
            json.dump(self.domain_array.tolist(), open(os.path.join(root_dir, 'domain_{}.json'.format(split)), 'w'))
            self.stats()
        else:
            self.domain_array = np.array(json.load(open(os.path.join(root_dir, 'domain_{}.json'.format(split)), 'r')))

        n_groups = len(self.group_names)
        self.group_counts = (np.arange(n_groups).reshape(n_groups, 1) == self.group_array).astype(np.float32).sum(1)
        n_domains = 2
        self.domain_counts = (np.arange(n_domains).reshape(n_domains, 1) == self.domain_array).astype(np.float32).sum(1)

        self.accum_losses = None
        self.weights = np.ones_like(self.y_array, dtype=np.float32)

        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.loader = default_loader

    def domain_split(self, split):
        np.random.seed(1)
        domain_array = np.ones_like(self.group_array)
        splits = {'train': {0: 65487, 1: 22880, 2: 0, 3: 22880},
                  'val':   {0: 8094,  1: 2874, 2: 0,  3: 2874},
                  'test':  {0: 7355,  1: 2480, 2: 0,  3: 2480}}
        split = splits[split]

        for d in range(len(self.group_names)):
            idx = np.where(self.group_array == d)[0]
            selected_idx = np.random.choice(idx, split[d], replace=False)
            domain_array[selected_idx] = 0

        return domain_array

    def stats(self):
        counts = [0,] * len(self.group_names)
        split_array = self.domain_array
        split_counts = {s: [0,] * len(self.group_names) for s in range(len(self.domain_counts))}

        for g, s in zip(self.group_array, split_array):
            split_count = split_counts[s]
            counts[g] += 1
            split_count[g] += 1

        for s in range(len(split_counts)):
            print('split: {}'.format(s))
            split_count = split_counts[s]
            for g, count in enumerate(counts):
                print('{}: {}/{}'.format(self.group_names[g], split_count[g], count))
        print('-' * 25)

    def get_groupname(self, c, y):
        return '-'.join([self.confounder_dict[c], self.y_dict[y]])

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img = self.loader(os.path.join(self.img_dir, self.filename_array[idx]))
        if self.transform is not None:
            img = self.transform(img)
        y = self.y_array[idx]
        g = self.group_array[idx]
        d = self.domain_array[idx]
        w = self.weights[idx]
        return img, y, g, d, w


def recompute(celeba_dataset, split_with_group, losses, beta):
    if split_with_group:
        split_array = celeba_dataset.group_array.copy()
        split_counts = celeba_dataset.group_counts
    else:
        split_array = celeba_dataset.domain_array.copy()
        split_counts = celeba_dataset.domain_counts

    if celeba_dataset.accum_losses is None:
        celeba_dataset.accum_losses = losses
    else:
        ema = 0.1
        celeba_dataset.accum_losses = celeba_dataset.accum_losses * (1 - ema) + losses * ema

    losses = celeba_dataset.accum_losses

    n_splits = len(split_counts)
    min_w = 0.1
    for s in range(n_splits):
        count = int(split_counts[s])
        idx = np.where(split_array == s)[0]
        idx_sorted = idx[np.argsort(losses[idx])[::-1]]
        cutoff_count = int((1 - min_w) * count * beta / (1 - min_w * beta))
        celeba_dataset.weights[idx] = min_w
        celeba_dataset.weights[idx_sorted[:cutoff_count]] = 1.0 / beta
        # leftover_mass = 1.0 - (frac[:cutoff_count].sum() / beta)
        # tiebreak_fraction = leftover_mass / frac[cutoff_count]  # check!
        # celeba_dataset.weights[idx_sorted[cutoff_count]] = max(tiebreak_fraction, 0.1)
