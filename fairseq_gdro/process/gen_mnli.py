import os
import sys
import numpy as np
import random
import string
import jsonlines
from collections import Counter

random.seed(1)
np.random.seed(1)

raw_dir = "/private/home/chuntinz/work/data/multinli/raw"
opt_dir = "/private/home/chuntinz/work/data/multinli/split_raw"
meta_path = os.path.join(raw_dir, "metadata_random.csv")
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)

split_dict = {
    'train': 0,
    'val': 1,
    'test': 2
}
reverse_split_dict = {v: k for k, v in split_dict.items()}

label_dict = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}
reverse_label_dict = {v: k for k, v in label_dict.items()}


def tokenize(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.lower()
    s = s.split(' ')
    return s


def check(s, label):
    negation_words_1 = ['nobody', 'nothing']  # Taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words_2 = ['no', 'never']
    def _read(tokens, marks):
        for token in tokens:
            if token in marks:
                return True
        return False
    if label == "neg1":
        return _read(tokenize(s), negation_words_1)
    else:
        return _read(tokenize(s), negation_words_2)


def read_json(path):
    data = []
    with jsonlines.open(path) as f:
        for line in f:
            sent1 = line['sentence1']
            sent2 = line['sentence2']
            label = line['gold_label']
            data.append((sent1, sent2, label))
    return data


def read_meta(path):
    data = []
    with open(path, "r") as fin:
        ii = 0
        for line in fin:
            if ii == 0:
                print(line)
            else:
                fields = line.strip().split(",")
                idx, label, neg1, neg2, split = fields
                data.append(list(map(int, fields)))
            ii += 1
    print("Total lines in meta data {}".format(len(data)))
    return data


def get_label(gold_label, neg1, neg2):
    double_negation = 0
    if neg1 == 0 and neg2 == 0:
        base = 0
    elif neg1 == 1:
        if neg2 == 1:
            double_negation = 1
        base = 1
    elif neg2 == 1:
        base = 2
    label = int(base * 3 + gold_label)
    assert 0 <= label < 9
    return label, double_negation


def resplit_all(data, meta_data):
    final_data = {"train": {k:[] for k in range(9)}, "val": [], "test":[]}
    double_negation = {"train": 0, "val": 0, "test": 0}
    for idx, label, neg1, neg2, split in meta_data:
        sent1, sent2, ll = data[idx]
        assert label_dict[ll] == label
        if neg1 == 1:
            assert check(sent2, "neg1")
        elif neg2 == 1:
            assert check(sent2, "neg2")
        finegrain_label, double_neg = get_label(label, neg1, neg2)
        double_negation[reverse_split_dict[split]] += double_neg
        if reverse_split_dict[split] == "train":
            final_data[reverse_split_dict[split]][finegrain_label].append((label, finegrain_label, sent1, sent2))
        else:
            final_data[reverse_split_dict[split]].append((label, finegrain_label, sent1, sent2))
    print("Double negation: ", double_negation)
    return final_data


def map_group(finegrain_label):
    if finegrain_label == 0 or finegrain_label == 1 or finegrain_label == 2:
        return 2
    elif finegrain_label == 3 or finegrain_label == 7 or finegrain_label == 8:
        return 0
    else:
        return 1


def resplit_train(train_data):
    final_train_num = {0: 0, 1:0, 2:0}
    final_train = []
    for finegrain_label in train_data.keys():
        group = map_group(finegrain_label)
        for label, finegrain_label, sent1, sent2 in train_data[finegrain_label]:
            final_train.append((label, finegrain_label, sent1, sent2, group))
        final_train_num[group] += len(train_data[finegrain_label])
        print("Finegrain label {} has {}".format(finegrain_label, len(train_data[finegrain_label])))
    for k, v in final_train_num.items():
        print("Group {} has {}".format(k, v))
    return final_train


def write_data(data, split, opt_path):
    sent1_path = os.path.join(opt_path, "{}.sent1".format(split))
    sent2_path = os.path.join(opt_path, "{}.sent2".format(split))
    label_path = os.path.join(opt_path, "{}.label".format(split))
    finegrain_label_path = os.path.join(opt_path, "{}.fg.labels".format(split))
    if split == "train":
        # noisy labels (imperfect partition)
        train_label_path = os.path.join(opt_path, "train.resplit.labels")

    all_fg = []
    with open(sent1_path, "w", encoding='utf-8') as fsent1, open(sent2_path, "w", encoding='utf-8') as fsent2, \
            open(label_path, "w", encoding='utf-8') as flabel, \
            open(finegrain_label_path, "w", encoding='utf-8') as ffg_label:
        if split == "train":
            ftrain_label = open(train_label_path, "w", encoding='utf-8')
        for fields in data:
            if split == "train":
                label, finegrain_label, sent1, sent2, train_label = fields
            else:
                label, finegrain_label, sent1, sent2 = fields
            all_fg.append(finegrain_label)
            fsent1.write(sent1.strip() + '\n')
            fsent2.write(sent2.strip() + '\n')
            flabel.write(reverse_label_dict[label].strip() + '\n')
            ffg_label.write(str(finegrain_label) + '\n')
            if split == "train":
                ftrain_label.write(str(train_label).strip() + '\n')
    print("{} {}".format(split, Counter(all_fg)))
    if split == 'train':
        ftrain_label.close()


train = read_json(os.path.join(raw_dir, "multinli_1.0_train.jsonl"))
dev = read_json(os.path.join(raw_dir, "multinli_1.0_dev_matched.jsonl"))
test = read_json(os.path.join(raw_dir, "multinli_1.0_dev_mismatched.jsonl"))

data = train + dev + test
print("Total combined data {}".format(len(data)))
meta_data = read_meta(os.path.join(raw_dir, "metadata_random.csv"))
resplit_data = resplit_all(data, meta_data)

write_data(resplit_data["val"], "valid", opt_dir)
write_data(resplit_data["test"], "test", opt_dir)

resplit_train_data = resplit_train(resplit_data["train"])
write_data(resplit_train_data, "train", opt_dir)