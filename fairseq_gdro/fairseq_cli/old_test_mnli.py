#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import os
import argparse
import logging
import math
import random
import sys

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def read_log(path, fout):
    def parse_line(line):
        fg_acc = dict()
        fields = line.strip().split("|")
        epoch = -1
        for field in fields:
            field = field.strip()
            if field.startswith("accuracy"):
                avg_acc = float(field.split()[-1].strip())
            elif field.startswith("fg_gacc"):
                fg_acc[field.split()[0].strip()] = float(field.split()[-1].strip())
            elif field.startswith("epoch"):
                epoch = int(field.split()[-1].strip().lstrip("0"))

        return fg_acc, avg_acc, epoch

    def write(fg_acc, avg_acc, split):
        fout.write("{} avg acc: \n".format(split))
        fout.write(" ".join([str(a) for a in avg_acc]) + "\n")
        fout.write("{} fine-grained acc: \n".format(split))
        lines = dict()
        for k, v in fg_acc.items():
            idx = int(k[7:])
            lines[idx] = k + " " + " ".join([str(a) for a in v]) + "\n"
        for idx in range(len(lines)):
            fout.write(lines[idx])
        fout.write("\n")

    def lt(l1, l2):
        l1 = sorted(l1)
        l2 = sorted(l2)
        idx = 0
        while idx < len(l1) and l1[idx] == l2[idx]:
            idx += 1
        if idx == len(l1):
            return False
        else:
            return l1[idx] < l2[idx]

    # read log and find early stopped checkpoint and change checkpoint path
    valid_fg_acc = defaultdict(list)
    test_fg_acc = defaultdict(list)
    train_fg_acc = defaultdict(list)
    valid_avg_acc = []
    train_avg_acc = []
    test_avg_acc = []
    max_worst_acc_valid = None
    best_epoch = 0

    with open(path) as fin:
        for line in fin:
            if "test on" in line or "valid on" in line or "train | epoch" in line:
                fg_acc, avg_acc, epoch = parse_line(line)
            if "test on" in line:
                for k, v in fg_acc.items():
                    test_fg_acc[k].append(v)
                test_avg_acc.append(avg_acc)
            elif "valid on" in line:
                for k, v in fg_acc.items():
                    valid_fg_acc[k].append(v)
                if best_epoch == 0 or lt(max_worst_acc_valid, fg_acc.values()):
                    max_worst_acc_valid = fg_acc.values()
                    best_epoch = epoch
                valid_avg_acc.append(avg_acc)
            elif "train | epoch" in line:
                for k, v in fg_acc.items():
                    train_fg_acc[k].append(v)
                train_avg_acc.append(avg_acc)

    fout.write("epoch {} with best worst group loss = {}\n".format(best_epoch, max_worst_acc_valid))
    if len(test_fg_acc) > 0:
        write(test_fg_acc, test_avg_acc, "test")
    if len(valid_fg_acc) > 0:
        write(valid_fg_acc, valid_avg_acc, "valid")
    if len(train_fg_acc) > 0:
        write(train_fg_acc, train_avg_acc, "train")
    return best_epoch


def main(args):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"

    save_dir = args.save_dir
    args.test_subset = "test"
    log_opt = open(os.path.join(save_dir, "test.log"), "w")
    best_epoch = read_log(os.path.join(save_dir, "log.txt"), log_opt)
    args.restore_file = os.path.join(save_dir, "checkpoint{}.pt".format(best_epoch))

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    task.load_dataset(args.test_subset, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.max_sentences
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    log_opt.write("Test on checkpoint {}\n".format(best_epoch))
    test_subsets = args.test_subset.split(",")
    stats = test(args, trainer, epoch_itr, test_subsets)
    fg_accs = dict()
    for key, value in stats.items():
        if key == "accuracy":
            log_opt.write("Accuracy {}\n".format(value))
        if key.startswith("fg_gacc"):
            idx = int(key[7:])
            fg_accs[idx] = value
    worst_acc = min(fg_accs.values())
    print("Worst_acc {}\n".format(worst_acc))
    log_opt.write("FG_acc {}\n".format(" ".join([str(fg_accs[idx]) for idx in range(len(fg_accs))])))
    log_opt.write("Worst_acc {}\n".format(worst_acc))
    log_opt.close()


def test(args, trainer, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"test on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = agg.get_smoothed_values()
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        return stats


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
