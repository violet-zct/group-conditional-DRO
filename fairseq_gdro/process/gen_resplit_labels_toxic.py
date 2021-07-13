import os
from collections import defaultdict

datasets = ["founta"]
root = "/private/home/chuntinz/work/data/hatespeech"


def read_tsv(path):
    i = 0
    data = []  # label, attribute, input
    labels = set()
    attributes = set()
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if i == 0:
                i += 1
                continue
            fields = line.strip().split("\t")
            i += 1
            data.append((fields[0], fields[1], fields[-1]))
            labels.add(fields[0])
            attributes.add(fields[1])
    # print("labels: ", labels)
    # print("attributes: ", attributes)
    return data


def map_to_finegrained_labels(label_id, attribute_id, num_attributes):
    return label_id * num_attributes + attribute_id


for dname in datasets:
    print(dname)
    if dname == "davidson":
        label_dict = {"offensive":0, "neither":1, "hate":2}
    else:
        label_dict = {"abusive":0, "spam":1, "normal":2, "hateful":3}

    attr_dict = {'other':0, 'aav':1, 'hispanic':2, 'white':3}
    raw_dir = os.path.join(root, dname, "raw")
    split_dir = os.path.join(root, dname, "split_raw")

    if not os.path.exists(split_dir):
        os.mkdir(split_dir)

    all_labels = defaultdict(int)
    for split in ["train"]:
        data = read_tsv(os.path.join(raw_dir, "{}.tsv".format(split)))
        with open(os.path.join(split_dir, "{}.resplit.labels".format(split)), "w", encoding="utf-8") as flabels:
            for label, att, sent in data:
                original_label = label.strip()
                label = label_dict[label.strip()]
                attid = attr_dict[att.strip()]
                flabels.write(str(attid) + "\n")
                all_labels[attid] += 1

    print(all_labels)