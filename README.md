# GNN-DSC180A

## GraphSMOTE
### dependencies
CPU
* python3
* ipdb
* pytorch1.0
* network 2.4
* scipy
* sklearn
### Command in Terminal to Run Code
##### Run Pretrain
python main.py --imbalance --dataset=amazon --setting='recon'
**in checkpoint file, change recon_300_False_0.5.pth to Pretrained.pth.pth, then run finetune**
##### Run Finetune
python main.py --imbalance --dataset=amazon --setting='newG_cls' --load=Pretrained.pth

## Improved GraphSmote
### dependencies
CPU
* python3
* ipdb
* pytorch1.0
* network 2.4
* scipy
* sklearn
### Command in Terminal to Run Code
##### Run Pretrain
python main.py --imbalance --dataset=amazon --setting='recon'
**in checkpoint file, change recon_300_False_0.5.pth to Pretrained.pth.pth, then run finetune**
##### Run Finetune
python main.py --imbalance --dataset=amazon --setting='newG_cls' --load=Pretrained.pth

# Baseline Models
### dependencies
Jupyter Notebook
* torch
* Numpy==1.21.5
* Pytorch==1.10.1
* DGL==0.8.2
* pygod == 1.1.0
* sklearn
* functools

### .ipynb files worked on JupiterNotebook
