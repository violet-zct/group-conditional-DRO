#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

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
import os

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(args):
    utils.import_user_module(args)

    args.no_epoch_checkpoints = True
    args.no_last_checkpoints = True

    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    if args.test_subset is not None:
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

    # fixme, hack
    if hasattr(trainer.criterion, "initialize"):
        trainer.criterion.initialize(getattr(task.datasets["train"], "num_splits", args.num_train_groups))
        trainer._criterion = trainer._criterion.to(device=trainer.device)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))
    if hasattr(checkpoint_utils.save_checkpoint, "best_epoch"):
        logger.info("best checkpoint = {}".format(checkpoint_utils.save_checkpoint.best_epoch))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = args.valid_subset.split(",")
    # test_subsets = args.test_subset.split(",") if args.test_subset is not None else None
    test_subsets = None
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

                # # for debug: to delete
                # valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop, valid_stats = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    if test_subsets is not None:
        test(args, trainer, epoch_itr, test_subsets)

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    trainer.resplit_train_epoch += 1
    # reweight instances
    if args.dynamic_group_dro and valid_stats is not None:
        group_weights = [stats['w{}'.format(ii)] for ii in range(trainer.get_criterion().n_train_groups)]
        max_w = max(group_weights)
        logger.info(" ".join(["w{} = {}".format(ii, group_weights[ii]) for ii in range(len(group_weights))]))
        # new reweighting criterion with worst-case valid performance
        # for language generation, acc is log-likelihood
        valid_group_acc = [(int(key.lstrip("fg_gacc")), valid_stats[key]) for key in valid_stats.keys() if key.startswith("fg_gacc")]
        worst_valid_acc = min([acc for _, acc in valid_group_acc])
        sorted_by_group_id = sorted(valid_group_acc, key=lambda tup: tup[0])
        group_acc = " ".join(["%d: %.3f" % (idx, acc if acc > 0 else -acc) for idx, acc in sorted_by_group_id])
        become_better = (trainer.worst_valid_acc is not None and worst_valid_acc > trainer.worst_valid_acc) or trainer.worst_valid_acc is None
        trainer.worst_valid_acc = worst_valid_acc if trainer.worst_valid_acc is None else max(worst_valid_acc, trainer.worst_valid_acc)
        trainer.bad_counts = 0 if become_better else trainer.bad_counts + 1
        logger.info("Valid group performance: {}".format(group_acc))
        logger.info("Better worst valid = {}, bad counts = {}, worst acc = {}".format(become_better, trainer.bad_counts, worst_valid_acc))

        '''if conservative: satisfy both conditions then update inner weights; else: as long as one of the conditions satisfies
        e.g. for every-epoch update: set --conservative=0, --resplit-epoch=-1, --resplit-patience=-1
        e.g. for update only when valid worst acc drops, set --conservative=1, --resplit-epoch=-1, --resplit-patience=0 
        '''
        if_update = (trainer.resplit_train_epoch > args.resplit_epoch and trainer.bad_counts > args.resplit_patience) \
            if args.conservative else \
            (trainer.resplit_train_epoch > args.resplit_epoch or trainer.bad_counts > args.resplit_patience)
        if if_update:
            if args.beta_cover_instances > 0:
                itr = trainer.get_valid_iterator("valid_train").next_epoch_itr(shuffle=False)
                train_losses = np.zeros(len(task.datasets["valid_train"]))
                counter = 0
                for sample in itr:
                    ind_losses, data_indices = trainer.valid_loss_step(sample)
                    train_losses[data_indices.cpu().numpy()] = ind_losses.cpu().numpy()
                    counter += 1
                    if counter % 100 == 0:
                        logger.info("Processed {} batches on each GPU".format(counter))
                torch.cuda.empty_cache()
                task.datasets[args.train_subset].resplit(epoch_itr.epoch, train_losses)
            else:
                task.datasets[args.train_subset].resplit(epoch_itr.epoch)
            trainer.criterion.reset_history()
            trainer.resplit_train_epoch = 0

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    do_save = (
        args.save_interval_updates > 0
        and num_updates > 0
        and num_updates % args.save_interval_updates == 0
        and num_updates >= args.validate_after_updates
    ) or (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or (args.validate_interval_updates > 0 and num_updates % args.validate_interval_updates == 0)
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    valid_stats = None
    if do_validate:
        valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    max_update = args.max_update or math.inf
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or trainer.get_num_updates() >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop, valid_stats


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
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
            prefix=f"valid on '{subset}' subset",
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
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses, stats


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    stats["worst_acc"] = min([stats[key] for key in stats.keys() if key.startswith("fg_gacc")])
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


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
