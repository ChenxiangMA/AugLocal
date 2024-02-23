import argparse
import os

import random
import shutil
import time
import warnings
from datetime import datetime
import logging
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import numpy as np
import pandas as pd

from utils.others import *

import networks.resnet
import networks.vgg

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/data/public/imagenet2012',
                    help='path to dataset, /datasets/imagenet,/data/public/imagenet2012')
parser.add_argument('--arch', default='resnet', type=str)
parser.add_argument('--net', default='resnet34', type=str)
parser.add_argument('--pre_train', default='', type=str, metavar='PATH',
                    help='path to the pre-train model (default: none)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_module_num', default=1, type=int,
                    help='number of local modules (1 refers to end-to-end training)')
parser.add_argument('--aux_net_config', default='unifSamp', type=str, help='[unifSamp, DGL]')
parser.add_argument('--aux_net_widen', default=1.0, type=float)

parser.add_argument('--save_path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--sync_bn', action='store_true', default=False, help='')
parser.add_argument('--aux_net_depth', default=1, type=int,
                    help='')
parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--layerwise_train', action='store_true', default=False, help='')


def main():
    args = parser.parse_args()
    if args.save_path == '':
        save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = save_path + args.name
    else:
        save_path = args.save_path
    args.use_ddp = True

    if args.use_ddp:
        from utils import distributed
        distributed.init_distributed_mode(args)

    if args.use_ddp:
        torch.distributed.barrier()
        if distributed.is_main_process():
            from pathlib import Path

            Path(save_path).mkdir(parents=True, exist_ok=True)
            # Logging settings
            setup_logging(os.path.join(save_path, 'log.txt'))
            logging.info('saving to:' + str(save_path))
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Logging settings
        setup_logging(os.path.join(save_path, 'log.txt'))
        logging.info('saving to:' + str(save_path))

    if args.use_ddp:
        is_cuda = True
        device = torch.device('cuda', args.gpu)
        set_random_seed(seed=args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')
        # Reproducibility
        reproducible_config(seed=args.seed, is_cuda=is_cuda)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform_train
    )

    if args.use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        assert args.batch_size % args.world_size == 0
        batch_size = args.batch_size // args.world_size
    else:
        train_sampler = None
        batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # dump args
    if distributed.is_main_process():
        with open(save_path + '/args.json', 'w') as fid:
            json.dump(args.__dict__, fid, indent=2)

    logging.info('args:' + str(args))

    model = eval('networks.' + args.arch + '.' + args.net)(local_module_num=args.local_module_num,
                                                           aux_net_config=args.aux_net_config,
                                                           aux_net_widen=args.aux_net_widen,
                                                           aux_net_depth=args.aux_net_depth,
                                                           layerwise_train=args.layerwise_train,
                                                           )
    logging.info(str(model))

    if args.pre_train:
        pre_train_model = torch.load(args.pre_train)
        model.load_state_dict(pre_train_model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    best_acc1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(device)
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # best_acc1 may be from a checkpoint from a different GPU
            # best_acc1 = best_acc1.to(device)

            # from collections import OrderedDict
            # dict = OrderedDict()
            state_dict = checkpoint['state_dict']
            for (key, value) in list(state_dict.items()):
                if key.startswith('module.'):
                    state_dict[key.replace("module.", "")] = value
                del state_dict[key]
            for k, v in state_dict.items():
                # print(k)
                # print(v.size())
                if 'weight_sigma' in k:
                    raise NotImplementedError
                    print(k)
                    state_dict[k] = v.view(1)

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for k, v in optimizer.state.items():  # key is Parameter, val is a dict {key='momentum_buffer':tensor(...)}
                if 'momentum_buffer' not in v:
                    continue
                optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'].cuda(args.gpu)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if is_cuda:
        model.to(device)

    if args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=True,
        )

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.evaluate:
        validate(val_loader, model, criterion_ce, args)
        return
    all_res = dict()
    for epoch in range(args.start_epoch, args.epochs):
        if args.use_ddp:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc1, train_loss, train_out_loss = train(train_loader, model, criterion_ce, optimizer, epoch, args)

        if not args.use_ddp or (args.use_ddp and distributed.is_main_process()):
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion_ce, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            all_res[str(epoch)] = [float(train_acc1), float(acc1), float(train_loss), float(train_out_loss)]

            pd.DataFrame(all_res.items()).to_csv(os.path.join(save_path, 'results.csv'), index=False)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=os.path.join(save_path, 'checkpoint.pth.tar'), save_path=save_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    output_losses = AverageMeter('OutLoss', ':.4e')
    top1 = [AverageMeter('Acc@1', ':6.2f') for _ in range(args.local_module_num)]
    top5 = [AverageMeter('Acc@5', ':6.2f') for _ in range(args.local_module_num)]

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, output_losses, top1[-1], top5[-1]],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()
        if args.layerwise_train:
            hidden_logits_record = []
            if args.net == 'resnet34' or args.net == 'resnet101':
                if args.net == 'resnet34':
                    layers = [1, 3, 4, 6, 3]  # resnet34
                elif args.net == 'resnet101':
                    layers = [1, 3, 4, 23, 3] # resnet101
                else:
                    raise NotImplementedError
                x = images
                for stage_i in (0, 1, 2, 3, 4):
                    for layer_i in range(layers[stage_i]):
                        x, loss, hidden_logits_record = model(x=x,
                                                              target=target,
                                                              criterion=criterion,
                                                              stage_i=stage_i,
                                                              layer_i=layer_i,
                                                              hidden_logits_record=hidden_logits_record)
                        if stage_i == 4 and layer_i + 1 == layers[stage_i]:
                            loss.backward()
                            optimizer.step()
                            output = x
                            output_loss = loss.item()
                        else:
                            with model.no_sync():
                                loss.backward()
            elif args.net == 'vgg13':
                x = images
                layer_i_list = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
                for layer_i in layer_i_list:
                    x, loss, hidden_logits_record = model(x=x,
                                                          target=target,
                                                          criterion=criterion,
                                                          stage_i=0, layer_i=layer_i,
                                                          hidden_logits_record=hidden_logits_record)
                    if layer_i == 13:
                        loss.backward()
                        optimizer.step()
                        output = x
                        output_loss = loss.item()
                    else:
                        with model.no_sync():
                            print(layer_i)
                            loss.backward()

        else:
            output, loss, hidden_logits_record, output_loss = model(x=images,
                                                                    target=target,
                                                                    criterion=criterion,
                                                                    )

            loss.backward()
            optimizer.step()

        module_i = 0
        for hidden_logits in hidden_logits_record:
            acc_hidden1, acc_hidden5 = accuracy(hidden_logits.data, target, topk=(1, 5))
            top1[module_i].update(acc_hidden1[0], hidden_logits.size(0))
            top5[module_i].update(acc_hidden5[0], hidden_logits.size(0))
            module_i += 1
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        output_losses.update(output_loss, target.size(0))
        top1[-1].update(acc1[0], target.size(0))
        top5[-1].update(acc5[0], target.size(0))
        # assert module_i + 1 == args.local_module_num

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print('max_mem={:.0f}MiB'.format(torch.cuda.max_memory_allocated() / 1e6))
        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
            progress.display(i)
            if (i + 1) == len(train_loader):
                out_string = '\n'
                for module_num in range(args.local_module_num):
                    out_string += 'layer [{0}], Prec@1 ({top1.avg:.3f}) ({top5.avg:.3f})\n'.format(module_num,
                                                                                                   top1=top1[
                                                                                                       module_num],
                                                                                                   top5=top5[
                                                                                                       module_num])
                out_string += ('mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(torch.cuda.memory_allocated() / 1e6,
                                                                           torch.cuda.max_memory_allocated() / 1e6))
                logging.info(out_string)

    return top1[-1].avg, losses.avg, output_losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or (i + 1) == len(val_loader):
                progress.display(i)

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_path='./'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        print('lr:')
        print(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




from PIL import ImageFilter





if __name__ == '__main__':
    main()
