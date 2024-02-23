import argparse
import logging
import os
import shutil
import time
import errno
import math
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils.others import *
import json
import pandas as pd
import random
import networks.resnet
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser(description='Implementation of AugLocal in PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: [cifar10|stl10|svhn]')
parser.add_argument('--model', default='resnet', type=str, help='resnet is supported currently')
parser.add_argument('--layers', default=0, type=int, help='total number of layers (have to be explicitly given!)')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='', type=str, help='name of experiment')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true', default=False,
                    help='whether to use cosine learning rate')
parser.add_argument('--local_module_num', default=1, type=int,
                    help='number of local modules (1 refers to end-to-end training)')
parser.add_argument('--aux_net_config', default='0c1f', type=str,
                    help='architecture of auxiliary networks for the local CE loss')
parser.add_argument('--aux_net_config_aux_loss', default='0c1f', type=str,
                    help='architecture of auxiliary networks for additional local loss')
parser.add_argument('--aux_net_widen', default=1.0, type=float,
                    help='widen factor of the auxiliary nets for the local CE loss')
parser.add_argument('--aux_net_widen_aux_loss', default=1.0, type=float,
                    help='widen factor of the auxiliary nets for additional local loss')
parser.add_argument('--aux_net_feature_dim', default=1024, type=int,
                    help='number of hidden features in auxiliary classifier')
parser.add_argument('--hidden_dim', default=512, type=int, help='number of hidden features in auxiliary classifier')
parser.add_argument('--hidden_dim_aux_loss', default=512, type=int,
                    help='number of hidden features in additional auxiliary net')
parser.add_argument('--ixx_1', default=0.0, type=float, )  # \lambda_1 for 1st local module
parser.add_argument('--ixy_1', default=1.0, type=float, )  # \lambda_2 for 1st local module
parser.add_argument('--ixx_2', default=0.0, type=float, )  # \lambda_1 for (K-1)th local module
parser.add_argument('--ixy_2', default=1.0, type=float, )  # \lambda_2 for (K-1)th local module
parser.add_argument('--epochs', default=160, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--rule', default='AugLocal', type=str, help='DGL, InfoPro, PredSim, AugLocal')
parser.add_argument('--save_path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--aux_net_depth', default=0, type=int, help='predefined depth of auxiliary networks in AugLocal')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--pyramid', default=False, action='store_true', help='enable pyramidal depth in AugLocal')
parser.add_argument('--pyramid_coeff', default=0.5, type=float, help='coefficient of pyramidal depth')
args = parser.parse_args()

if args.save_path == '':
    save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = save_path + args.name + '_' + str(args.seed)
else:
    save_path = args.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)
# Logging settings
setup_logging(os.path.join(save_path, 'log.txt'))
logging.info('saving to:' + str(save_path))

if args.rule == 'AugLocal':
    args.aux_net_config = 'unifSamp'
    args.local_module_num = args.local_module_num - args.aux_net_depth
elif args.rule == 'DGL':
    args.aux_net_config = 'DGL'
elif args.rule == 'PredSim':
    args.aux_net_config = 'PredSim'
elif args.rule == 'InfoPro':
    args.aux_net_config = 'InfoPro'
    args.aux_net_config_aux_loss = '2c'

# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': args.epochs,
        'batch_size': args.batch_size if args.dataset in ['cifar10', 'svhn'] else 128,
        'initial_learning_rate': args.lr if args.dataset in ['cifar10', 'svhn'] else 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': args.wd,
    }
}


def main():
    set_random_seed(seed=args.seed)
    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []

    class_num = args.dataset in ['cifar10', 'sl10', 'svhn'] and 10 or 100

    if 'cifar' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
    else:
        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}

    if 'cifar' in args.dataset:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        image_size = 32
    elif 'stl' in args.dataset:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(96, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize])
        image_size = 96
    elif 'svhn' in args.dataset:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=2),
             transforms.ToTensor(),
             normalize])
        image_size = 32
    else:
        raise NotImplementedError

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('/datasets/' + args.dataset, download=True,
                                                transform=transform_train,
                                                **kwargs_dataset_train),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, drop_last=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('/datasets/' + args.dataset, transform=transform_test, download=True,
                                                **kwargs_dataset_test),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)

    with open(save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info('args: ' + str(args))

    # create model
    if args.model == 'resnet':
        model = eval('networks.resnet.resnet' + str(args.layers)) \
            (local_module_num=args.local_module_num,
             batch_size=training_configurations[args.model]['batch_size'],
             image_size=image_size,
             dataset=args.dataset,
             class_num=class_num,
             wide_list=(16, 16, 32, 64),
             dropout_rate=args.droprate,
             aux_net_config=args.aux_net_config,
             aux_net_config_aux_loss=args.aux_net_config_aux_loss,
             aux_net_widen=args.aux_net_widen,
             aux_net_widen_aux_loss=args.aux_net_widen_aux_loss,
             aux_net_feature_dim=args.aux_net_feature_dim,
             hidden_dim=args.hidden_dim,
             hidden_dim_aux_loss=args.hidden_dim_aux_loss,
             rule=args.rule,
             aux_net_depth=args.aux_net_depth,
             is_pyramid=args.pyramid,
             paramid_coeff=args.pyramid_coeff,
             )
    else:
        raise NotImplementedError
    model.eval()

    logging.info(str(model))

    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        # np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0
    # all_res = pd.DataFrame()
    # writer = SummaryWriter(save_path, purge_step=start_epoch)
    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):
        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train_acc, loss_output, loss_x, loss_y = train(train_loader, model, optimizer, epoch, None)

        # evaluate on validation set
        prec1 = validate(val_loader, model, epoch, None)
        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('test_acc', prec1, epoch)
        logging.info('train acc {:.4f} test acc {:.4f}'.format(train_acc, prec1))
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # all_res = pd.concat(
        #     [all_res, pd.DataFrame({str(epoch): np.array([float(train_acc), float(prec1), float(loss_output)])})],
        #     axis=1)
        # if args.rule in ['InfoPro', 'PredSim']:
        #     all_res = pd.concat(
        #         [all_res, pd.DataFrame({str(epoch) + 'loss_x': np.array([lossx.ave for lossx in loss_x])})],
        #         axis=1)
        # all_res = pd.concat(
        #     [all_res, pd.DataFrame({str(epoch) + 'loss_y': np.array([lossy.ave for lossy in loss_y])})],
        #     axis=1)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_acc': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        #     'val_acc': val_acc,
        # }, is_best, filename=os.path.join(save_path, 'checkpoint.pth.tar'), save_path=save_path)

        logging.info('Best accuracy: ' + str(best_prec1))
    # all_res.to_csv(os.path.join(save_path, 'results.csv'), index=False)


def train(train_loader, model, optimizer, epoch, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    losses_ixx = [AverageMeter() for _ in
                  range(args.local_module_num - 1)]  # additional loss term in PredSim and InfoPro
    losses_ixy = [AverageMeter() for _ in range(args.local_module_num - 1)]  # CE loss
    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (x, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)

        optimizer.zero_grad()
        output, loss, _, loss_ixx_list, loss_ixy_list = model(img=x,
                                                              target=target,
                                                              ixx_1=args.ixx_1,
                                                              ixy_1=args.ixy_1,
                                                              ixx_2=args.ixx_2,
                                                              ixy_2=args.ixy_2,
                                                              )

        optimizer.step()
        module_i = 0
        # for hidden_logits in hidden_logits_record:
        #     acc_hidden = accuracy(hidden_logits.data, target, topk=(1,))[0]
        #     top1[module_i].update(acc_hidden.item(), hidden_logits.size(0))
        #     module_i += 1

        # if args.rule in ['InfoPro', 'PredSim']:
        #     for i_loss, loss_x in enumerate(loss_ixx_list):
        #         losses_ixx[i_loss].update(loss_x.data.item(), target.size(0))

        # for i_loss, loss_y in enumerate(loss_ixy_list):
        #     losses_ixy[i_loss].update(loss_y.data.item(), target.size(0))

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), target.size(0))
        top1.update(prec1.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) == len(train_loader):
            # print(discriminate_weights)

            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})'.format(
                epoch, i + 1, train_batches_num, batch_time=batch_time,
                loss=losses, top1=top1))
            string += (
                'mem={:.0f}MiB, max_mem={:.0f}MiB, max_mem={:.0f}MiB \n'.format(torch.cuda.memory_allocated() / 1e6,
                                                                                torch.cuda.max_memory_allocated() / 1e6,
                                                                                torch.cuda.max_memory_reserved() / 1e6))
            # for module_num in range(args.local_module_num):
            #     string += 'layer [{0}], Prec@1 {top1.value:.3f} ({top1.ave:.3f})\n'.format(module_num,
            #                                                                                top1=top1[module_num])
            # writer.add_scalar(f'layer [{module_num}]', top1[module_num].ave, epoch)

            logging.info(string)
            # print(weights)
    return top1.ave, losses.ave, losses_ixx, losses_ixy


def validate(val_loader, model, epoch, writer):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output, loss, _, _, _ = model(img=input, target=target, )

        module_i = 0
        # for hidden_logits in hidden_logits_record:
        #     acc_hidden = accuracy(hidden_logits.data, target, topk=(1,))[0]
        #     top1[module_i].update(acc_hidden.item(), hidden_logits.size(0))
        #     module_i += 1

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\n'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    # for module_num in range(args.local_module_num):
    #     string += 'layer [{0}], Prec@1 {top1.value:.3f} ({top1.ave:.3f})\n'.format(module_num, top1=top1[module_num])
    # writer.add_scalar(f'layer [{module_num}]', top1[module_num].ave, epoch)
    logging.info(string)

    val_acc.append(top1.ave)

    return top1.ave


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_path='./'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate'] \
                                    * (1 + math.cos(
                    math.pi * epoch / training_configurations[args.model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (
                                            11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate'] \
                                    * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()
