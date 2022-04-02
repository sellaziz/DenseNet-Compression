# DenseNet-Compression
MicroNet Compression Challenge on DenseNet for CIFAR-10 (for Deep Learning Compression Course at School)
# Our Strategy
First we tried to tinker with the hyperparameters to get the lower MicroNet Score and get an accuracy over 90% (while keeping 1 or 2% to allow us other ways of compressing the network)
# Our Final Results 

![Results](resources\score_comp.png)
x is MobileNet and MobileNetv2, other are DenseNet (with growth rate as legend)

![Best Results](resources\score_comp_best.png)




| Model       | Growth Rate | Sparsity | Accuracy | Encoding (bits) | Groups | Distillation | Score   | Comment                                   | Params | Name                     |
| ----------- | ----------- | -------- | -------- | --------------- | ------ | ------------ | ------- | ----------------------------------------- | ------ | ------------------------ |
| DenseNet    | 4           | 0.651    | 90.04    | 32              | x      | x            | 0.02524 |                                           | 70922  | dense_gr4_sp65           |
| DenseNet    | 4           | 0        | 90.61    | 32              | x      | x            | 0.0335  |                                           | 70922  | dense_gr4                |
| DenseNet    | 6           | 0        | 91.83    | 32              | x      | x            | 0.0729  |                                           | 150610 | dense_gr6                |
| DenseNet    | 8           | 0        | 93.26    | 32              | x      | x            | 0.127   |                                           | 259786 | dense_gr8                |
| MobileNet   | 0           | 0        | 86.52    | 32              | x      | x            | 0.1481  |                                           | 303173 | mbn                      |
| MobileNetv2 | 0           | 0        | 94.14    | 32              | x      | x            | 0.3994  |                                           | 158045 | mbnv2                    |
| DenseNet    | 4           | 0        | 91.57    | 32              | x      | o            | 0.0335  |                                           | 70922  | dense_gr4_dist           |
| DenseNet    | 4           | 0        | 90.47    | 32              | x      | x            | 0.0286  | Depthwise-Decomposition                   | 66282  | dense_gr4_dws            |
| DenseNet    | 4           | 0        | 88.47    | 32              | 2      | x            | 0.0226  |                                           | 35461  | dense_gr4_grp2           |
| DenseNet    | 4           | 0        | 90.12    | 32              | 2      | o            | 0.0226  |                                           | 35461  | dense_gr4_grp2_dist      |
| DenseNet    | 4           | 0.3017   | 90.04    | 32              | 2      | x            | 0.0207  |                                           | 35461  | dense_gr4_grp2_sp30      |
| DenseNet    | 4           | 0        | 90.54    | 32              | 2      | o            | 0.0205  | Depthwise-Decomposition, groups sur conv1 | 66282  | dense_gr4_grp2_dws       |
| DenseNet    | 4           | 0.32     | 90.02    | 16              | 2      | o            | 0.018   | Depthwise-Decomposition, groups sur conv1 | 66282  | dense_gr4_grp2_dws_sp32  |
| DenseNet    | 3           | 0        | 88.51    | 32              | x      | x            | 0.0219  |                                           | 34631  | dense_gr3                |
| DenseNet    | 4           | 0        | 90.25    | 8               | x      | o            | 0.01706 | distill pré ptq quant int8                | 70922  | dense_gr4_dist_int8      |
| DenseNet    | 4           | 0.651    | 90.04    | 8               | x      | o            | 0.0167  | local quant int8 and sparse               | 70922  | dense_gr4_dist_int8_sp65 |
| DenseNet    | 3           | 0        | 90.03    | 32              | x      | o            | 0.0219  |                                           | 34631  | dense_gr3                |
| DenseNet    | 2           | 0        | 84.60    | 32              | x      | x            | 0.0102  |                                           | 16389  | dense_gr2                |
