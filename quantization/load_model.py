import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import numpy as np
# from models import *
import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# parser.add_argument('--pruned', '-p', action='store_true',
#                     help='load pruned model')
parser.add_argument('--ckpt', default="./chekpoints/weights.pth", type=str,
                    help='model path')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--gr', default=4, type=int,
                    help='Growth Rate Parameter for DenseNet')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Used device: ",device)

# Load checkpoint.
print('==> Resuming from checkpoint..')
ckpt="./checkpoints/DenseNet_gr4_conv1-grp2_conv2-dw-sep_db-arch-6-12-24-16_distill_pruned_32_percent.state_dict"
assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
state_dict = torch.load(args.ckpt, map_location=device)
net.load_state_dict(checkpoint['net'])
# if args.pruned:
#     print('==> Building model..')
#     from models.densenet_dw2 import DenseNet, Bottleneck
#     net = DenseNet121()
#     net = net.to(device)
#     prune.identity(net.linear, 'weight')
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             prune.identity(m, 'weight')
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             prune.remove(m,'weight')

# COLORED
def print_sparsity_colored(net):
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

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class_names=classes
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

# from minicifar import minicifar_train, minicifar_test, train_sampler, valid_sampler, n_classes_minicifar
def prepare_data_loaders(data_path):

    print('==> Preparing data..')
    augment=False
    batch_size=32
    if augment:
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
    cifar_path=data_path
    
    trainset = datasets.CIFAR10(root=cifar_path, train=True, download=False,
                                transform=transform_train)
    data_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=8)

    testset = datasets.CIFAR10(root=cifar_path, train=False, download=False,
                            transform=transform_test)
    data_loader_test = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)

    return data_loader, data_loader_test

def evaluation(model, test_loader, criterion, n_class, half=False): 

    test_loss = 0.0
    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))

    model.eval()
    # for data, label in test_loader:
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        data = inputs.to(device=device)
        label = labels.to(device=device)
        if half:
            data=data.half()
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, label)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(label.data.view_as(pred)))
        for i in range(len(label)):
            digit = label.data[i]
            class_correct[digit] += correct[i].item()
            class_total[digit] += 1

    test_loss = test_loss/len(test_loader.sampler)
    print('test Loss: {:.6f}\n'.format(test_loss))
    for i in range(n_class):
        print('test accuracy of %s: %2d%% (%2d/%2d)' % (class_names[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
    return 100. * np.sum(class_correct) / np.sum(class_total)
criterion = nn.CrossEntropyLoss()
# testloader = DataLoader(minicifar_test, batch_size=32)
data_loader, testloader = prepare_data_loaders(data_path)

evaluation(net, testloader, criterion, 10)

from torch.profiler import profile, record_function, ProfilerActivity

def estimate_throughput(model, testloader, half=False):
    # inputs = torch.randn(5, 3, 224, 224).cuda()
    it = iter(testloader)
    first = next(it)
    inputs = first[0].to(device)
    if half:
        inputs=inputs.half()
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

estimate_throughput(net, testloader, half=False)
net.half()
evaluation(net, testloader, criterion, 10, half=True)
estimate_throughput(net, testloader, half=True)