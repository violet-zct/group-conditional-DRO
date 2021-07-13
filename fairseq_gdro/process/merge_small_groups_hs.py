import os
from collections import defaultdict
root = "/private/home/chuntinz/work/data/hatespeech/founta/bin"

stat = defaultdict(int)
labels = []
with open(os.path.join(root, "test.fg.labels")) as fin:
    for line in fin:
        stat[line.strip()] += 1
        labels.append(line.strip())

new_map = dict()
idx = 0
start_idx = 1
for key, count in stat.items():
    if count <= 100:
        new_map[key] = idx
    else:
        new_map[key] = start_idx
        start_idx += 1

print("before = {}, after = {}".format(len(stat), len(new_map)))
with open(os.path.join(root, "test.labels"), "w") as fout:
    for label in labels:
        fout.write(str(new_map[label]) + "\n")