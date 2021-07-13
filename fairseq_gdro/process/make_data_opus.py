import os
import numpy as np
import sys
import random
import shutil

np.random.seed(1)
random.seed(1)

# make debug data
domains = ['law', 'medical', 'it', 'koran', 'subtitles']
root_dir = "/private/home/chuntinz/work/data/opus_wmt14/bpe" # under bpe run this
opt_dir = "/private/home/chuntinz/work/data/opus_wmt14/mix_bpe/opus_v5"
root_opt = "/private/home/chuntinz/work/data/opus_wmt14/mix_bpe/"
# ratios = [200, 100, 100, 300, 50]
cutoff = 125

if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)


def is_overlong(l1, l2):
    if len(l1.split()) < cutoff and len(l2.split()) < cutoff:
        return False
    return True


def is_overlong_valid(l1, l2):
    valid_cut_off = 100
    if len(l1.split()) < valid_cut_off and len(l2.split()) < valid_cut_off:
        return False
    return True


def is_overshort(l1, l2):
    if len(l1.split()) < 4 or len(l2.split()) < 4:
        return True
    return False


def is_misaligned(l1, l2):
    if len(l1.split()) == 0 or len(l2.split()) == 0:
        return True
    if len(l1.split()) * 1.0 / len(l2.split()) >= 3 or len(l2.split()) * 1.0 / len(l1.split()) >= 3:
        return True
    return False


def read_data(f1, f2, filter=False):
    data = []
    with open(f1, encoding='utf-8') as fin1, open(f2, encoding='utf-8') as fin2:
        for l1, l2 in zip(fin1, fin2):
            if filter and is_overlong_valid(l1.strip(), l2.strip()):
                continue
            else:
                data.append((l1.strip(), l2.strip()))
    return data


def main():
    train_data = dict()
    valid_data = dict()

    train_fsrc, train_ftgt = "train.en-de.en", "train.en-de.de"
    valid_fsrc, valid_ftgt = "valid.en-de.en", "valid.en-de.de"
    test_fsrc, test_ftgt = "test.en-de.en", "test.en-de.de"

    for d in domains:
        p1, p2 = os.path.join(root_dir, d, "train.en-de.en"), os.path.join(root_dir, d, "train.en-de.de")
        train_data[d] = read_data(p1, p2)

        p1, p2 = os.path.join(root_dir, d, "valid.en-de.en"), os.path.join(root_dir, d, "valid.en-de.de")
        valid_data[d] = read_data(p1, p2, filter=True)


    mix_valid_data = []
    for d in domains:
        random.shuffle(valid_data[d])
        for l1, l2 in valid_data[d][:1000]:
            mix_valid_data.append((l1, l2))

    with open(os.path.join(opt_dir, train_fsrc), 'w', encoding='utf-8') as fout1, open(os.path.join(opt_dir, train_ftgt), 'w', encoding='utf-8') as fout2, open(os.path.join(opt_dir, "train.labels"), 'w', encoding='utf-8') as fout3:
        for ii, d in enumerate(domains):
            cutoff_num = 0
            misaligned = 0
            total = 0
            tokens = 0
            random.shuffle(train_data[d])

            sep_opt_dir = os.path.join(root_opt, d)
            sep_src = open(os.path.join(sep_opt_dir, train_fsrc), 'w', encoding='utf-8')
            sep_tgt = open(os.path.join(sep_opt_dir, train_ftgt), 'w', encoding='utf-8')
            sep_valid_src = open(os.path.join(sep_opt_dir, valid_fsrc), 'w', encoding='utf-8')
            sep_valid_tgt = open(os.path.join(sep_opt_dir, valid_ftgt), 'w', encoding='utf-8')

            for l1, l2 in train_data[d]:
                if is_overlong(l1, l2):
                    cutoff_num += 1
                    continue
                if is_misaligned(l1, l2):
                    misaligned += 1
                    continue

                if d == "subtitles" and is_overshort(l1, l2):
                    cutoff_num += 1
                    continue

                total += 1
                fout1.write(l1 + '\n')
                fout2.write(l2 + '\n')
                fout3.write(str(ii) + '\n')

                sep_src.write(l1 + "\n")
                sep_tgt.write(l2 + "\n")
                tokens += len(l1.strip().split())
                if d == "subtitles" and total >= 500000:
                    break
            print(d, tokens, misaligned, cutoff_num, total)

            for l1, l2 in valid_data[d]:
                sep_valid_src.write(l1 + "\n")
                sep_valid_tgt.write(l2 + "\n")

            shutil.copyfile(os.path.join(root_dir, d, test_fsrc), os.path.join(sep_opt_dir, test_fsrc))
            shutil.copyfile(os.path.join(root_dir, d, test_ftgt), os.path.join(sep_opt_dir, test_ftgt))

            sep_src.close()
            sep_tgt.close()
            sep_valid_src.close()
            sep_valid_tgt.close()

    with open(os.path.join(opt_dir, valid_fsrc), 'w', encoding='utf-8') as fout1, open(os.path.join(opt_dir, valid_ftgt), 'w', encoding='utf-8') as fout2:
        for l1, l2 in mix_valid_data:
            fout1.write(l1 + '\n')
            fout2.write(l2 + '\n')

    for d in domains:
        shutil.copyfile(os.path.join(root_dir, d, test_fsrc), os.path.join(opt_dir, "{}.{}".format(d, test_fsrc)))
        shutil.copyfile(os.path.join(root_dir, d, test_ftgt), os.path.join(opt_dir, "{}.{}".format(d, test_ftgt)))



main()
