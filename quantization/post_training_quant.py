# Based On Quantization Tutorial : https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html


import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
from torch.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
# from torch.jit.quantized import QuantizedLinear, Quan
# Setup warnings
import warnings
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
print("date and time =", dt_string)	
import sys
sys.stdout = open('ptq_'+dt_string+'.log','wt')
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)


# from torchvision.models.resnet import resnet18
from torch.quantization import get_default_qconfig, quantize_jit, default_weight_only_qconfig, QConfig
# from models.densenet import DenseNet, Bottleneck
from models.densenet_dw2 import DenseNet, Bottleneck, Transition


##################### Custom ##################################
class_names = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# class_names=classes[0:n_classes_minicifar]
device= torch.device('cpu')
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
###########################################################


################## Define Helper Functions and Prepare Dataset ############################################

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5

def load_model(model_file):
    model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=4)

    device = torch.device('cpu')
    state_dict = torch.load(model_file, map_location=device)
    # own_state = model.state_dict()
    # for name, param in state_dict.items():
    #     if name not in own_state:
    #             continue
    #     if isinstance(param, nn.Parameter):
    #         # backwards compatibility for serialized parameters
    #         param = param.data
    #     own_state[name].copy_(param)
    model = state_dict['net']
    model.to(torch.device('cpu'))
    model = model.module.to(device)
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

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

data_path = '.~/data/cifar10'
saved_model_dir = 'checkpoints/'
float_model_file = 'DenseNet_gr4_distille_sparcity_65'

# train_batch_size = 30
# eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to("cpu")

# float_model = float_model.module.to("cpu")


# float_model = float_model.module.to("cpu") # for DataParallel modules
# float_model = float_model.module.module.to("cpu")
float_model = float_model.module.module.module.module.module.to("cpu")
import torch.nn.utils.prune as prune
parameters_to_prune = [(m, 'weight') for m in float_model.modules() if isinstance(m, nn.modules.Conv2d) or isinstance(m, nn.modules.Linear)]
for module,_ in parameters_to_prune:
    prune.remove(module,'weight')
# prune.remove(float_model, 'weight')

# print(sum([param.is_cuda for param in float_model.parameters()]*1))
float_model.eval()

# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)
################## ################################# ############################################

################## Set model to eval mode ############################################
model_to_quantize.eval()
################## ################################# ############################################

################## Specify how to quantize the model with qconfig_dict ############################################
qconfig = get_default_qconfig("fbgemm")

qconfig_dict = {"": qconfig}
print(qconfig_dict)
# Trick to prevent quantization of some modules
# prepare_custom_config_dict = {
#     # option 1
#     "non_traceable_module_name": [
#         "trans3", 
#         "dense1", 
#         "dense2"
#     ],
#     # option 2
#     "non_traceable_module_class": []
# }
# print(prepare_custom_config_dict)
print(float_model_file)
################## ################################# ############################################

################## Prepare the Model for Post Training Static Quantization ############################################
prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
# prepared_model = prepare_fx(model_to_quantize, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)
################## ################################# ############################################

################## Calibration ############################################
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)
calibrate(prepared_model, data_loader_test)  # run calibration on sample data
################## ################################# ############################################

################## Calibration ############################################
quantized_model = convert_fx(prepared_model)
print(quantized_model)
################## ################################# ############################################

################## Evaluation ############################################
print("Size of model before quantization")
print_size_of_model(float_model)
print("Size of model after quantization")
print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serilaization] Evaluation accuracy on test dataset: %2.4f, %2.4f"%(top1.avg, top5.avg))

# evaluation(quantized_model, data_loader_test, criterion, 10)

fx_graph_mode_model_file_path = saved_model_dir + "densenet_fx_graph_mode_quantized.pth"

# this does not run due to some erros loading convrelu module:
# ModuleAttributeError: 'ConvReLU2d' object has no attribute '_modules'
# save the whole model directly
# torch.save(quantized_model, fx_graph_mode_model_file_path)
# loaded_quantized_model = torch.load(fx_graph_mode_model_file_path)

# save with state_dict
# torch.save(quantized_model.state_dict(), fx_graph_mode_model_file_path)
# import copy
# model_to_quantize = copy.deepcopy(float_model)
# prepared_model = prepare_fx(model_to_quantize, {"": qconfig})
# loaded_quantized_model = convert_fx(prepared_model)
# loaded_quantized_model.load_state_dict(torch.load(fx_graph_mode_model_file_path))

# save with script
torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)
loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)

top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.4f, %2.4f"%(top1.avg, top5.avg))
# evaluation(loaded_quantized_model, data_loader_test, criterion, 10)

################## ################################# ############################################
fused = fuse_fx(float_model)

conv1_weight_after_fuse = fused.conv1.weight[0]
conv1_weight_after_quant = quantized_model.conv1.weight().dequantize()[0]
print(torch.max(abs(conv1_weight_after_fuse - conv1_weight_after_quant)))

fuse_iter=iter(fused.named_modules())
quant_iter=iter(quantized_model.named_modules())
## Get Quantization Noise for each Layers, doesn't work for partial quantization
while True:
    try:
        # get the next item
        fuse_name,fuse_elem = next(fuse_iter)
        quant_name,quant_elem = next(quant_iter)
        while True:
            try:
                while type(fuse_elem).__name__=="ReLU" or type(fuse_elem).__name__=="BatchNorm2d" or type(fuse_elem).__name__=="ConvReLU2d":
                    fuse_name,fuse_elem = next(fuse_iter)
                    # print(fuse_name)
                break
            except StopIteration:
                # if StopIteration is raised, break from loop
                break
        print(type(fuse_elem).__name__, type(quant_elem).__name__)
        print(f"Names : {fuse_name}, Quant : {quant_name}, ")
        if "Conv" in type(fuse_elem).__name__ and "Conv" in type(quant_elem).__name__:
            fuse_mod=fuse_elem.weight
            quant_mod=quant_elem.weight().dequantize()
            max_diff=torch.max(abs(fuse_mod - quant_mod))
            # print(fuse_elem, quant_elem)
            print(f"Names : {fuse_name}, Quant : {quant_name}, max diff : {max_diff}")
        # do something with element
    except StopIteration:
        # if StopIteration is raised, break from loop
        break



from torch.profiler import profile, record_function, ProfilerActivity
inputs = torch.randn(32, 3, 32, 32)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        float_model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        quantized_model(inputs)
        # loaded_quantized_model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



####################################################################################################
# Histogram Plot to analyze the repartition of the weights
# import matplotlib.pyplot as plt
# begining, ending= -5, 2.5
# chunksize=2**8
# stepsize=(ending-begining)/chunksize
# bins=np.linspace(begining,ending,num=chunksize)
# sub_hist = torch.zeros(chunksize)
# min=100
# max=0
# # import re
# from torch.nn import Conv2d, Linear
# for name, m in quantized_model.named_modules():
#     if type(m).__name__=='Conv2d':
#         try:
#             clipped_weight=m.weight()
#         except:
#             print("no weight")
#         try:
#             clipped_weight=m.weight().dequantize()
#             sub_hist_temp  = torch.histc(clipped_weight, bins=chunksize, min=begining,max=ending)
#             sub_hist=torch.vstack((sub_hist,sub_hist_temp.detach().cpu()))
#             max= np.maximum(torch.max(clipped_weight).cpu().detach().numpy(), max)
#             min= np.minimum(torch.min(clipped_weight).cpu().detach().numpy(), min)
#         except Exception as e: 
#             print(e)
#             print("no weight")
# print(min, max)


# unique_weights=torch.Tensor()
# unique_weights = unique_weights.to(device)
# prev_len=unique_weights.shape[0]
# for name, m in quantized_model.named_modules():
#     if type(m).__name__=='Conv2d' :
#         try:
#             clipped_weight=m.weight().dequantize()
#             for filter in m.weight().dequantize():
#                 tmp_unique = torch.unique(filter)
#             # clipped_weight=m.weight
#             tmp_unique = torch.unique(clipped_weight)
#             # print(tmp_unique.shape, end= " ")
#             unique_weights=torch.unique(torch.cat((unique_weights,tmp_unique),0))
#         except Exception as e:
#             print("no weight", e)
# print(f"{unique_weights.shape=}")

# hist_sum = sub_hist.sum(axis=0)
# thresh=0.01
# # idx=(np.abs(bins)-thresh<0) # To Fine-tune
# # print(f"Number of parameter less than {thresh} : {(hist_sum[idx].sum()/hist_sum.sum())*100:.2f}% ({int(hist_sum[idx].sum())}/{int(hist_sum.sum())})")
# # hist_sum[idx]=0
# plt.bar(bins,hist_sum,width=stepsize)
# plt.title("Histogram of Weights, Original")
# plt.savefig('hist_full_no0.1.png')
####################################################################################################



################## Specify how to quantize the model with qconfig_dict ############################################
################## ################################# ############################################

