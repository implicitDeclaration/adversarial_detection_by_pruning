
# Introduction

Code to the paper You Can’t Fool All The Models: Detect Adversarial Samples via Pruning Models and Adversarial sample detection via channel pruning

The implementation of this artifact is based on pytorch 1.6 with python 3.7. 

# Code Structure

This artifact includes four independent modules.

- model Generation (structured and unstructured)
- Adversarial Sample Generation (attacks)
- Label Change Rate and AUC over adversarial samples (lcr_auc)
- Adversarial Sample Detection (detect)


# Useage

### 1. Model Generation
structured model Generation:

```
python structured.py --config ./config/resnet18-stru.yaml --multigpu 0 --ori_model [path of your original model] --cfg resnet18 --random_rule [random_pretrain or l1_pretrain]
```
unstructured model Generation 

```
python unstructured.py --config ./config/resnet18-unstru.yaml --multigpu 0 --trainer [default or lottery]
```
### 2. Adversarial Samples Generation

fgsm attack:

```
python attacks/craft_adversarial_img.py --config ../config/resnet18-unstru.yaml --multigpu 0 --pretrained [path of your original model] --attackType fgsm
```
### 3. Label Change Rate and AUC Calculation
mutated testing on normal sample:

```
python lcr_auc/mutated_testing.py --config ./config/resnet18-unstru.yaml --multigpu 0 --prunedModelsPath [path of your pruned models] --testType normal > normal.log
```
lcr and auc calculation on normal sample:

```
python lcr_auc/lcr_auc_analysis.py --config ./config/resnet18-unstru.yaml --multigpu 0 --is_adv False --maxModelsUsed 100 --lcrSavePath [path to save lcr result] --logPath [directory of your normal.log]
```
When finished, we can get lcr of normal samples(threshold for adversarial sample detection). The value is equal to: avg_lcr+99%confidence


### 4. Adversarial Sample Detection
Adversarial Samples detection:

```
python detect/adv_detect.py --config ./config/resnet18-unstru.yaml --multigpu 0 --prunedModelsPath [path of your pruned models] --testSamplesPath [path of your adversarial samples] --threshold [lcr of normal samples] --testType adv
```

# Reference
- [dgl-prc/m_testing_adversatial_sample](https://github.com/dgl-prc/m_testing_adversatial_sample)
- [allenai/hidden-networks](https://github.com/allenai/hidden-networks)
- [lmbxmu/abcpruner](https://github.com/lmbxmu/abcpruner)

If our works helped you, consider cite
'''bash
@article{wang2021you,
  title={You Can’t Fool All the Models: Detect Adversarial Samples via Pruning Models},
  author={Wang, Renxuan and Chen, Zuohui and Dong, Hui and Xuan, Qi},
  journal={IEEE Access},
  volume={9},
  pages={163780--163790},
  year={2021},
  publisher={IEEE}
}
'''











 
