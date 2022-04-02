"""
Generate an histogram on weights values
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import numpy as np
from models import *
import os
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Used device: ",device)
print('==> Building model..')
net = DenseNet121()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
ckpt="./checkpoint/retrain_pruned_0.8_30_epoch.pth"
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# checkpoint = torch.load('./checkpoint/ckpt.pth')
checkpoint = torch.load(ckpt)
# prune.identity(net.linear, 'weight')
for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # print("pruning babeeeeee")
        prune.identity(m, 'weight')
net.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # print("pruning babeeeeee")
    #    print(m.weight_mask)
       prune.remove(m,'weight')

begining, ending= -1.5, 1.5
chunksize=100
stepsize=(ending-begining)/chunksize
bins=np.linspace(begining,ending,num=chunksize)
sub_hist = torch.zeros(chunksize)
min=100
max=0
for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        sub_hist_temp  = torch.histc(m.weight, bins=chunksize, min=begining,max=ending)
        sub_hist=torch.vstack((sub_hist,sub_hist_temp.detach().cpu()))
        max= np.maximum(torch.max(m.weight).cpu().detach().numpy(), max)
        min= np.minimum(torch.min(m.weight).cpu().detach().numpy(), min)


hist_sum = sub_hist.sum(axis=0)
idx=(np.abs(bins)-0.15<0) # To Fine-tune
hist_sum[idx]=0
plt.bar(bins,hist_sum,width=stepsize)
plt.title("Histogram of Weights, Pruned Model (80%)")
plt.savefig('hist.png')

