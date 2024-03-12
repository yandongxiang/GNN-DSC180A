# GNN-DSC180A

## Requirements
pip install -r /path/to/requirements.txt

## GraphSMOTE
### dependencies
CPU
* argparse
* numpy
* scipy
* matplotlib
* torch
* torch-geometric
* dgl
* networkx
* scikit-learn
* ipdb
* pygod

### Command in Terminal to Run Code
#### Run Pretrain
python main.py --imbalance --dataset=amazon --setting='recon'

**In checkpoint file, change "recon_300_False_0.5.pth" to "Pretrained.pth.pth", then run finetune**
#### Run Finetune
python main.py --imbalance --dataset=amazon --setting='newG_cls' --load=Pretrained.pth

## Improved GraphSmote
### dependencies
CPU
* argparse
* numpy
* scipy
* matplotlib
* torch
* torch-geometric
* dgl
* networkx
* scikit-learn
* ipdb
* pygod

### Command in Terminal to Run Code
#### Run Pretrain
python main.py --imbalance --dataset=amazon --setting='recon'

**In checkpoint file, change recon_300_False_0.5.pth to Pretrained.pth.pth, then run finetune**
#### Run Finetune
python main.py --imbalance --dataset=amazon --setting='newG_cls' --load=Pretrained.pth

# Visualization Generation
### dependencies

# Baseline Models
### dependencies
* torch
* Numpy==1.21.5
* Pytorch==1.10.1
* DGL==0.8.2
* pygod == 1.1.0
* sklearn
* functools
