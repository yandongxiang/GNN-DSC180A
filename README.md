# Detecting Fraud With Oversampling Techniques and Sparsity Contraints

**Project Overview**
Fraud detection is prevalent now more than ever due to the massive surge in the usage of online platforms. Many techniques exist to combat fraud; however, they often fail to capture the imbalanced class in data involving fraudulent activities. Our research contributes to the study of such concern with a model that harnesses the strengths of many existing models. We propose a solution that utilizes a combination of oversampling techniques and sparsity constraints to balance and predict fraud data.


**Structure**

```python
Project
├── README.md
├── visualization_generation.py
├── requirements.txt
├── Baseline Models
    ├── GAT Baseline.py
    ├── GCN Baseline.py
    ├── GIN Baseline.py
    └── Fraud Sage-ROC.py
├── GraphSmote-main
    ├── data
    ├── data_load.py
    ├── main.py
    ├── models.py
    └── utils.py
└── GraphSmote-main-improved
    ├── data
    ├── data_load.py
    ├── main.py
    ├── models.py
    └── utils.py
```


**Project Usage**

1. Clone the repo to your local machine
2. Open your terminal and cd into the directory of the cloned repo
3. On the terminal, run ```pip install -r requirements.txt``` to obtain necessary packages to run the code.
4. Next, cd into the designated folder you want to run. And then run ```python (name of the py file).py```. Explained more in detail below.

## Baseline Models

### Dependencies
* torch
* Numpy==1.21.5
* Pytorch==1.10.1
* DGL==0.8.2
* pygod == 1.1.0
* sklearn
* functools

### Terminal Command to Run Code
run designated model
Ex: ```python GAT Baseline.py```


## GraphSMOTE

### Terminal Command to Run Code
#### Run Pretrain
```python main.py --imbalance --dataset=amazon --setting='recon'```

**In checkpoint file, change "recon_300_False_0.5.pth" to "Pretrained.pth.pth", then run finetune**

#### Run Finetune
```python main.py --imbalance --dataset=amazon --setting='newG_cls' --load=Pretrained.pth```

## Improved GraphSmote

### Terminal Command to Run Code

#### Run Pretrain
```python main.py --imbalance --dataset=amazon --setting='recon'```

**In checkpoint file, change recon_300_False_0.5.pth to Pretrained.pth.pth, then run finetune**

#### Run Finetune
```python main.py --imbalance --dataset=amazon --setting='newG_cls' --load=Pretrained.pth```

# Generate Visualizations

```python visualization_generation.py```
