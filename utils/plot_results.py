"""
Plots graphs for pruning comparison, unstructured
"""
import numpy as np
import matplotlib.pyplot as plt
import json
 
# Opening JSON file
with open('results_pruning_l1.txt') as file:
    results_pruning_l1 = json.load(file)
    file.close()
with open('results_pruning_rd.txt') as file:
    results_pruning_rd = json.load(file)
    file.close()
l1_p_rate=list()
l1_p_val=list()
for key, val in results_pruning_l1.items():
    l1_p_rate.append(float(key))
    l1_p_val.append(float(val))
rd_p_rate=list()
rd_p_val=list()
for key, val in results_pruning_rd.items():
    rd_p_rate.append(float(key))
    rd_p_val.append(float(val))

# train_losses_sched = np.load('train_densenet_lr_0.1_True.npy', mmap_mode='r')
# valid_losses_sched = np.load('test_densenet_lr_0.1_True.npy', mmap_mode='r')

# plt.plot(train_losses_lr_001)
# plt.plot(train_losses_lr_01)
# plt.plot(train_losses_lr_1)
# plt.plot(train_losses_sched)
# plt.legend(['lr=0.001', 'lr=0.01','lr=0.1', 'Scheduler'])
# plt.title("Evolution of the Loss by epoch (different lr)")
# plt.xlabel("epoch")
# plt.ylabel('Value')
# plt.savefig('train_loss.png')

# plt.plot(valid_losses_lr_001)
# plt.plot(valid_losses_lr_01)
# plt.plot(valid_losses_lr_1)
# plt.plot(valid_losses_sched)
# plt.legend(['lr=0.001', 'lr=0.01','lr=0.1', 'Scheduler'])
# plt.title("Evolution of the Loss by epoch (different lr)")
# plt.xlabel("epoch")
# plt.ylabel('Value')
# plt.savefig('valid_loss.png')

plt.plot(rd_p_rate, rd_p_val)
plt.plot(l1_p_rate, l1_p_val)
# for i in range(len(rd_p_rate)):
#     plt.scatter(rd_p_rate[i], rd_p_val[i])
#     plt.annotate(str(rd_p_val[i]), (rd_p_rate[i], rd_p_val[i]))
#     plt.scatter(l1_p_rate[i], l1_p_val[i])
#     plt.annotate(str(l1_p_val[i]), (l1_p_rate[i], l1_p_val[i]))
plt.scatter(rd_p_rate[0], rd_p_val[0])
plt.annotate(str(rd_p_val[0]), (rd_p_rate[0], rd_p_val[0]))
plt.scatter(l1_p_rate[0], l1_p_val[0])
plt.annotate(str(l1_p_val[0]), (l1_p_rate[0], l1_p_val[0]))
plt.legend(['Random','L1 norm'])
plt.title("Evolution of the Accuracy by pruning Rate")
plt.xlabel("Pruning Rate")
plt.ylabel('Accuracy')
plt.savefig('accuracy_pruning_unstructured.png')