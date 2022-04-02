import os
import csv
import torch
import torch.nn as nn
# from models import *
# from models.densenet import DenseNet121
from models.densenet import DenseNet, Bottleneck, Transition
from torchsummary import summary

our_quant=32
par_quant_ratio=1. if our_quant==32 or our_quant==16 else (our_quant/32)
ops_quant_ratio=(our_quant/32)

def count_conv2d(m, x, y):
    x = x[0] # remove tuple
    fin = m.in_channels
    fout = m.out_channels
    sh, sw = m.kernel_size
    # print("###########################################################################################################################")
    # for attr in dir(m):
    #     try:
    #         print("########################", attr, "###########################")
    #         print(getattr(m, attr))
    #         try: 
    #             print(getattr(m, attr)())
    #             try: 
    #                 print(list(getattr(m, attr)()))
    #             except Exception as e:
    #                 print()
    #         except Exception as e:
    #             print()
    #     except Exception as e:
    #         print()
    print(m.parent, m.level.item())
    if m.level.item() in [2, 3]:
        factor=4
    else:
        factor=2
    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul/factor # FP16
    ops = (kernel_mul + kernel_add)/m.groups + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    #Nice Formatting
    # print("{:<10}: S_c={:<4}, F_in={:<4}, F_out={:<4}, P={:<5}, params={:<10}, operations={:<20}".format("Conv2d",sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, params={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple

    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div
    # total_ops=total_ops*par_quant_ratio


    m.total_ops += torch.Tensor([int(total_ops)])
    #Nice Formatting
    # print("{:<10}: S_c={:<4}, F_in={:<4}, F_out={:<4}, P={:<5}, params={:<10}, operations={:<20}".format("Batch norm",'x',x.size(1),'x',x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # print("Batch norm: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("ReLU: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),0,int(total_ops)))



def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    # total_ops=total_ops*par_quant_ratio

    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("AvgPool: S={}, F_in={}, P={}, params={}, operations={}".format(m.kernel_size,x.size(1),x.size()[2:].numel(),0,int(total_ops)))

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features/2
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    # total_ops=total_ops*par_quant_ratio

    print("Linear: F_in={}, F_out={}, params={}, operations={}".format(m.in_features,m.out_features,int(m.total_params.item()),int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    print ("Sequential: No additional parameters  / op")

# custom ops could be used to pass variable customized ratios for quantization
counter=0
def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        global counter
        if len(list(m.children())) > 0: 
            if isinstance(m, nn.Sequential): counter+=1
            if isinstance(m, Transition): 
                for name, m in m.named_modules():
                    # try:
                    #     m.parent
                    # except Exception as e:
                    #     print(e)
                    m.register_buffer('parent', torch.tensor(1)) # 1 is trans
            print(counter)
            return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))
        m.register_buffer('level', torch.tensor(counter))
        try:
            m.parent
        except:
            m.register_buffer('parent', torch.tensor(0)) # 0 is other

        for p in m.parameters():
            print(m.level.item())
            if m.level.item() in [2, 3]:
                factor=4
            else:
                factor=2
            m.total_params += torch.Tensor([p.numel()]) / factor # Division Free quantification
        
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params

    return total_ops, total_params

def main():
    # Resnet18 - Reference for CIFAR 10
    ref_params = 5586981
    ref_flops  = 834362880
    # WideResnet-28-10 - Reference for CIFAR 100
    # ref_params = 36500000
    # ref_flops  = 10490000000

    # model = DenseNet121()
    # model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=10) #densenet_cifar
    # # print(model)
    # summary(model, (3, 32,32))
    # growth_rates= [2,4,6,8,10,12,16,32]
    growth_rates= [4]
    db_architectures= [[6,12,24,16]]
                       #[6,12,16,16],
                       #[6, 8,16,16],
                       #[6,12,24,8]]
    logname='scores_gr_group2.csv'

    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['growth rate', 'score', 'score_params', 'score_flops','params', 'flops'])
    for gr in growth_rates:
        for db_arch in db_architectures:
            print(" DenseNet ".center(100,"#"))
            print(f"{gr=}, {db_arch=}")
            model = DenseNet(Bottleneck, db_arch, growth_rate=gr)
            # model = DenseNet121()
            flops, params = profile(model, (1,3,32,32))
            flops, params = flops.item(), params.item()

            score_flops = flops / ref_flops
            score_params = (params / ref_params)*(1-0.651)
            score = score_flops + score_params
            print("Flops: {}, Params: {}".format(flops,params))
            print("Score flops: {} Score Params: {}".format(score_flops,score_params))
            print("Final score: {}".format(score))

            
            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([gr, score, score_params, score_flops, params,flops])

if __name__ == "__main__":
    main()