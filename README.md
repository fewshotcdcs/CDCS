# CDCS

This is the PyTorch implementation for the following ICSE 2022 paper:

**Title**: [Cross-Domain Deep Code Search with Few-Shot Meta Learning](https://arxiv.org/abs/2201.00150)

## Introduction

This repo provides the code for reproducing the experiments in [Cross-Domain Deep Code Search with Few-Shot Meta Learning](https://arxiv.org/abs/2201.00150).
CDCS is a novel approach for domain-specific code search. CDCS employs a transfer learning framework where an initial program representation model is pre-trained on a large corpus of common programming languages (such as Java and Python), and is further adapted to domain-specific languages such as Solidity and SQL. 

Paper link: https://arxiv.org/abs/2109.00859

## Dataset

### Pre-training & Meta learning

CDCS pre-trained on Python and Java data from CodeBERT(https://github.com/microsoft/CodeBERT).
You can use the following command to download the preprocessed training and validation dataset: 

```
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo  
```

We use only positive samples for pre-training and use the entire set of pairs for meta learning.

We provide the train_valid data in this repo.

### Fine-tuning

CDCS fine-tuned on Solidty(https://zenodo.org/record/4587089#.YEog9-gzYuV) and SQL(https://github.com/taoyds/spider) data. 

We provide the processed data in this repo and the processing details are in our paper.

## Dependency
* pip install torch
* pip install transformers
* https://higher.readthedocs.io/en/latest/

## Script
As the model is based on Roberta, we provide the meta training file maml.py in this repo, which including the implementation details of MAML.
Other model implementation files can refer to codesearch task in CodeBERT.


For example, if you want to run the meta learning, you can simple run:
```
python maml.py \
--model_type roberta \
--task_name codesearch \
--do_meta_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--max_seq_length 256\
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 2 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models/$lang  \
--model_name_or_path $pretrained_model
```

If you want to run fine-tuning, you can replace the `--do_meta_train` with `--do_train`

