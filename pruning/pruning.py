from __future__ import print_function

import argparse
import csv
import os
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.utils.prune as prune

import models
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="DenseNet121", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='pruning_gr_4', type=str, help='name of run')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
torch.cuda.empty_cache()
if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_path='../lab2_aziz_12_02/lab2/data/cifar10'
trainset = datasets.CIFAR10(root=cifar_path, train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = datasets.CIFAR10(root=cifar_path, train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)

# Model
# if args.resume:
if True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
    #                         + str(args.seed))
    checkpoint = torch.load('./checkpoint/ckpt.acc_90_group_distillDataParallel_0_42db_arch6_12_24_16_gr_4')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if True:
    # Load checkpoint.
    print('==> Resuming from checkpoint MASTER..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('./checkpoint/ckpt.t7DataParallel_0_42db_arch6_12_24_16_gr_4')
    checkpoint = torch.load('./checkpoint/ckptacc915.gr4_distilleDataParallel_0_42db_arch6_12_24_16_gr_4')
    #checkpoint = torch.load('./checkpoint/ckpt.t70_0')
    master = checkpoint['net']

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        outputs_master= master(inputs)

        ditillation_loss = divergence_loss_fn(
            F.softmax(outputs_master , dim=1),
            F.softmax(outputs , dim=1)
        )

        loss = 0.5*criterion(outputs, targets) + abs(ditillation_loss)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets.data).cpu().sum().float()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # progress_bar(batch_idx, len(trainloader),
        #              'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
        #                 100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),
                        'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total,
                            correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    # if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
    #     checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.prunedgroupgr4_pruneRate' + str(prune_rate)+ args.name + '_'+ str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

parameters_to_prune=()
for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        parameters_to_prune=parameters_to_prune+((m, 'weight'),)
# print(parameters_to_prune)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_sparcity():
    global_sum=0
    glob_elem=0
    denseblock_counter=1
    for m in net.modules():
        if isinstance(m, nn.Sequential):
            print(f"DenseBlock {denseblock_counter}")
            denseblock_counter+=1
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            val=100. * float(torch.sum(m.weight == 0))/ float(m.weight.nelement())
            color=bcolors.BOLD
            if val <50:
                color=bcolors.OKGREEN
            elif 50<val<75:
                color=bcolors.OKBLUE
            elif val>75:
                color=bcolors.FAIL
            print(
                "Sparsity in {0}: ".format(type(m).__name__)+color +
                "{0:.2f}%".format(100. * float(torch.sum(m.weight == 0))/ float(m.weight.nelement())
                )+bcolors.ENDC
            )
            global_sum+=torch.sum(m.weight == 0)
            glob_elem+=m.weight.nelement()
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                global_sum
            )
            / float(
                glob_elem
            )
        )
    )
    return

def prunage(rate):
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=rate,
    )
    return


print(" PRUNNED NETWORK ".center(100,"#"))

sparcity=0
prune_rate=0.05
logname = ('results/log_prunedgroup_pruneRate'+str(prune_rate) + net.__class__.__name__ + '_' + args.name + '_'+ str(args.seed) + '.csv')
for i in range (100):
    prunage(prune_rate)
    sparcity=1-(1-prune_rate)**(i+1)
    #optimizer = optim.Adam(net.parameters())
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                       weight_decay=args.decay)
    best_acc=89.99
    #logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'+ str(args.seed) +'_sparcity' +str(int(sparcity*1000)/1000) + '.csv')
    if not os.path.exists(logname):
            with open(logname, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                    'test loss', 'test acc'])
    for epoch in range(50):
        train_loss, reg_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc])
        if test_acc>=90:
            break
    if test_acc<90:
        break

print_sparcity()

# for epoch in range(start_epoch, args.epoch):
#     train_loss, reg_loss, train_acc = train(epoch)
#     test_loss, test_acc = test(epoch)
#     #adjust_learning_rate(optimizer, epoch)
#     with open(logname, 'a') as logfile:
#         logwriter = csv.writer(logfile, delimiter=',')
#         logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
#                             test_acc])

