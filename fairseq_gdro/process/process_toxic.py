import os
from collections import OrderedDict

datasets = ["davidson", "founta"]
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

    all_labels = dict()
    fg_label_counts = dict()
    for split in ["train", "valid", "test"]:
        data = read_tsv(os.path.join(raw_dir, "{}.tsv".format(split)))
        with open(os.path.join(split_dir, "{}.sent0".format(split)), "w", encoding="utf-8") as ftxt, \
            open(os.path.join(split_dir, "{}.label".format(split)), "w", encoding="utf-8") as flabel, \
            open(os.path.join(split_dir, "{}.fg.labels".format(split)), "w", encoding="utf-8") as ffglabel:
            for label, att, sent in data:
                ftxt.write(sent.strip() + "\n")
                original_label = label.strip()
                label = label_dict[label.strip()]
                attid = attr_dict[att.strip()]
                flabel.write(str(label) + "\n")
                fglabel = map_to_finegrained_labels(label, attid, len(attr_dict))

                key = "{}-{}".format(original_label, att)
                if key in all_labels:
                    assert all_labels[key] == fglabel
                    if split == "train":
                        fg_label_counts[key] += 1
                else:
                    all_labels[key] = fglabel
                    if split == "train":
                        fg_label_counts[key] = 1

                ffglabel.write(str(fglabel) + "\n")
    print(all_labels)
    sorted_counts = OrderedDict({k: v for k, v in sorted(fg_label_counts.items(), key=lambda item: item[1])})
    print(sorted_counts)
