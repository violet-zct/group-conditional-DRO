import os
import sys
import gc

root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(root_path)

from argparse import ArgumentParser
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.celeba import CelebADataset, recompute
from loss import ERMLoss, DROEGLoss, DROGreedyLoss
from utils import AverageMeter


def parse_args():
    parser = ArgumentParser(description='CelebA')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50'], required=True)
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--batch_steps', type=int, default=1, metavar='N',
                        help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
    parser.add_argument('--eval_batch_size', type=int, default=512, metavar='N', help='input batch size for eval (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: None)')
    parser.add_argument('--run', type=int, default=1, metavar='N', help='number of runs for the experiment')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--lr', type=float, help='learning rate', required=True)
    parser.add_argument('--lr_eg', type=float, default=0.01, help='learning rate for exponential decay DRO')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for greedy group DRO')
    parser.add_argument('--beta', type=float, default=0.1, help='beta for greedy group DRO')
    parser.add_argument('--eg_normalize', action='store_true', default=False)
    parser.add_argument('--lr_decay', choices=['milestone', None], default=None, help='Decay rate of learning rate')
    parser.add_argument('--milestone', type=int, nargs='+', default=[80, 120], help='Decrease learning rate at these epochs.')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight for l2 norm decay')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--data_path', help='path for data file.', required=True)
    parser.add_argument('--model_path', help='path for saving model file.', required=True)

    parser.add_argument('--loss', choices=['erm', 'dro_eg', 'dro_greedy'])
    parser.add_argument('--reweight', action='store_true', default=False)
    parser.add_argument('--recompute', type=int, default=0)
    parser.add_argument('--group_split', choices=['confounder', 'domain'], default='domain', help='the type of splitting groups')

    return parser.parse_args()


def logging(info, logfile=None):
    print(info)
    if logfile is not None:
        print(info, file=logfile)
        logfile.flush()


def get_optimizer(lr, parameters, momentum, lr_decay, decay_rate, milestone, weight_decay):
    optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    opt = 'momentum={:.1f}, wd={:.1e}'.format(momentum, weight_decay)
    if lr_decay == 'milestone':
        opt = opt + 'lr decay={} {}, decay rate={:.3f}, '.format(lr_decay, milestone, decay_rate)
        scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=decay_rate)
    else:
        scheduler = None

    return optimizer, scheduler, opt


def setup(args):
    data_path = args.data_path

    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    args.log = open(os.path.join(model_path, 'log.run{}.txt'.format(args.run)), 'w')

    args.cuda = torch.cuda.is_available()
    random_seed = args.seed
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)

    logging("Args: " + str(args), args.log)

    trainset = CelebADataset(data_path, split='train',
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(
                                     (224, 224),
                                     scale=(0.7, 1.0),
                                     ratio=(1.0, 4 / 3),
                                     interpolation=2),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ]))
    valset = CelebADataset(data_path, split='val',
                           transform=transforms.Compose([
                               transforms.CenterCrop(178),
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))
    testset = CelebADataset(data_path, split='test',
                           transform=transforms.Compose([
                               transforms.CenterCrop(178),
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))

    logging('Data size: training: {}, val: {}, test: {}'.format(len(trainset), len(valset), len(testset)))

    pretrained = not args.train_from_scratch
    n_classes = 2
    n_grpups = 4
    n_domains = 2
    if args.arch == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=pretrained)
    else:
        raise ValueError('unknown neural architecture: {}'.format(args.arch))

    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)

    split_with_confounder = args.group_split == 'confounder'
    if args.loss == 'erm':
        model = ERMLoss(model, n_grpups)
    elif args.loss == 'dro_eg':
        assert args.reweight, 'DRO with exponential decay requires data re-sampling.'
        model = DROEGLoss(model, n_grpups, n_domains, args.lr_eg, 
                          split_with_group=split_with_confounder,
                          normalize=args.eg_normalize)
    elif args.loss == 'dro_greedy':
        assert not args.reweight, 'DRO with greedy requires no data re-sampling.'
        model = DROGreedyLoss(model, n_grpups, n_domains, args.alpha,
                              split_with_group=split_with_confounder)
    else:
        raise ValueError('unknown loss type: {}'.format(args.loss))

    model.to(device)
    args.device = device

    return args, (trainset, valset, testset), model


def init_dataloader(args, trainset, valset, testset):
    if args.recompute > 0:
        assert args.loss == 'dro_greedy'
        train_eval_loader = DataLoader(trainset, batch_size=args.eval_batch_size, shuffle=False,
                                       num_workers=args.workers, pin_memory=True)
    else:
        train_eval_loader = None

    if args.reweight:
        if args.group_split == 'confounder':
            group_weights = len(trainset) / trainset.group_counts
            weights = group_weights[trainset.group_array]
        else:
            assert args.group_split == 'domain'
            domain_weights = len(trainset) / trainset.domain_counts
            weights = domain_weights[trainset.domain_array]
        sampler = WeightedRandomSampler(weights, len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    return train_loader, train_eval_loader, val_loader, test_loader


def eval(args, eval_loader, model):
    model.eval()

    accum_loss = AverageMeter()
    accum_acc = AverageMeter()
    n_groups = 4
    accum_group_accs = [AverageMeter() for _ in range(n_groups)]
    accum_group_losses = [AverageMeter() for _ in range(n_groups)]

    device = args.device
    if args.cuda:
        torch.cuda.empty_cache()

    for x, y, g, d, w in eval_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        g = g.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        loss, acc, group_losses, group_accs, group_counts = model(x, y, g, d, w)

        accum_loss.update(loss.item(), x.size(0))
        accum_acc.update(acc.item(), x.size(0))
        for i in range(n_groups):
            accum_group_accs[i].update(group_accs[i].item(), group_counts[i].item())
            accum_group_losses[i].update(group_losses[i].item(), group_counts[i].item())

    return accum_loss.avg, accum_acc.avg, \
           [accum_group_losses[i].avg for i in range(n_groups)], \
           [accum_group_accs[i].avg for i in range(n_groups)]


def compute_loss(args, eval_loader, model):
    model.eval()
    device = args.device
    if args.cuda:
        torch.cuda.empty_cache()

    losses = []

    for x, y, _, _, _ in eval_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = model.compute_loss(x, y)
        losses.append(loss)

    return torch.cat(losses)


def train(args, train_loader, num_train, model, optimizer):
    model.train()
    start_time = time.time()

    accum_loss = AverageMeter()
    accum_acc = AverageMeter()
    n_groups = 4
    accum_group_accs = [AverageMeter() for _ in range(n_groups)]
    accum_group_losses = [AverageMeter() for _ in range(n_groups)]
    num_back = 0
    device = args.device

    if args.cuda:
        torch.cuda.empty_cache()

    for step, (x, y, g, d, w) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        g = g.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        loss, acc, group_losses, group_accs, group_counts = model(x, y, g, d, w)

        accum_loss.update(loss.item(), x.size(0))
        accum_acc.update(acc.item(), x.size(0))
        for i in range(n_groups):
            accum_group_accs[i].update(group_accs[i].item(), group_counts[i].item())
            accum_group_losses[i].update(group_losses[i].item(), group_counts[i].item())

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            log_info = '[{}/{} ({:.0f}%)] loss: {:.4f}, acc: {:.2f}%, '.format(
                accum_loss.count, num_train, 100. * accum_loss.count / num_train, accum_loss.avg, accum_acc.avg)
            log_info += ', '.join(['g{} ({:.3f}, {:.2f}%)'.format(i, accum_group_losses[i].avg, accum_group_accs[i].avg) for i in range(n_groups)])
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)

    log_info = 'Average loss: {:.4f}, acc: {:.2f}%, '.format(accum_loss.avg, accum_acc.avg)
    log_info += ', '.join(['g{} ({:.3f}, {:.2f}%)'.format(i, accum_group_losses[i].avg, accum_group_accs[i].avg) for i in range(n_groups)])
    log_info += ', time: {:.1f}s'.format(time.time() - start_time)
    logging(log_info, args.log)


def main(args):
    args, (trainset, valset, testset), model = setup(args)

    logging('# of Parameters: %d' % sum([param.numel() for param in model.parameters()]), args.log)

    train_loader, train_eval_loader, val_loader, test_loader = init_dataloader(args, trainset, valset, testset)

    epochs = args.epochs
    log = args.log

    opt = 'sgd'
    momentum = args.momentum
    lr_decay = args.lr_decay
    decay_rate = args.decay_rate
    milestone = args.milestone
    weight_decay = args.weight_decay

    optimizer, scheduler, opt_param = get_optimizer(args.lr, model.parameters(), momentum,
                                                    lr_decay=lr_decay, decay_rate=decay_rate,
                                                    milestone=milestone, weight_decay=weight_decay)

    n_groups = 4
    best_robust_acc = 0
    patient = 0
    for epoch in range(1, epochs + 1):
        lr = args.lr if scheduler is None else scheduler.get_last_lr()[0]
        logging('Epoch: {}/{} ({}, lr={:.5f}, {})'.format(epoch, epochs, opt, lr, opt_param), log)
        train(args, train_loader, len(trainset), model, optimizer)
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            loss, acc, group_losses, group_accs = eval(args, val_loader, model)
            robust_acc = min(group_accs)
            eval_test = False
            if robust_acc > best_robust_acc:
                eval_test = True
                best_epoch = epoch
                best_robust_acc = robust_acc
                patient = 0
            else:
                patient += 1
            log_info = 'val  loss: {:.4f}, acc: {:.2f}%, '.format(loss, acc)
            log_info += ', '.join(['g{} ({:.3f}, {:.2f}%)'.format(i, group_losses[i], group_accs[i]) for i in range(n_groups)])
            log_info += ', robust: {:.2f}%, | best epoch: {} ({:.2f}%), patient: {}'.format(robust_acc, best_epoch, best_robust_acc, patient)
            logging(log_info, log)

            if eval_test:
                loss, acc, group_losses, group_accs = eval(args, test_loader, model)
                robust_acc = min(group_accs)
                log_info = 'test loss: {:.4f}, acc: {:.2f}%, '.format(loss, acc)
                log_info += ', '.join(['g{} ({:.3f}, {:.2f}%)'.format(i, group_losses[i], group_accs[i]) for i in range(n_groups)])
                log_info += ', robust: {:.2f}%'.format(robust_acc)
                logging(log_info, log)

            if args.loss == 'dro_eg':
                ws_info = ', '.join(['w{}={:.3f}'.format(i, w.item()) for i, w in enumerate(model.adv_probs)])
                logging("EG weights: {}".format(ws_info), logfile=log)
            elif args.loss == 'dro_greedy':
                ws_info = ', '.join(['w{}={:.3f}'.format(i, w.item()) for i, w in enumerate(model.h_fun)])
                logging("Group weights: {}".format(ws_info), logfile=log)
                gloss_info = ', '.join(['g{}={:.3f}'.format(i, gs.item()) for i, gs in enumerate(model.sum_losses)])
                logging("Group losses:  {}".format(gloss_info), logfile=log)

            if args.recompute > 0:
                logging('***************Re-Compute at the end of epoch {}*****************'.format(epoch), log)
                losses = compute_loss(args, train_eval_loader, model)
                recompute(trainset, args.group_split == 'confounder', losses.cpu().numpy(), args.beta)
                model.reset_loss()


if __name__ == "__main__":
    args = parse_args()
    main(args)

