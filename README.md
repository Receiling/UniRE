# UniRE
Source code for ["UniRE: A Unified Label Space for Entity Relation Extraction."](https://aclanthology.org/2021.acl-long.19/), ACL2021.

It is based on our [NERE](https://github.com/Receiling/NERE) toolkit.

## Requirements
* `python`: 3.7.6
* `pytorch`: 1.8.1
* `transformers`: 4.2.2
* `configargparse`: 1.2.3
* `bidict`: 0.20.0
* `fire`: 0.3.1

## Datasets
We provide scripts and instructions for processing three datasets (ACE2004,ACE2005,SciERC) in the [`data/`](https://github.com/Receiling/UniRE/tree/master/data).

## Training
### ACE2004
```bash
python entity_relation_joint_decoder.py \
    --config_file config.yml \
    --save_dir ckpt/ace2004_bert \
    --data_dir data/ACE2004/fold1 \
    --fine_tune \
    --device 0
```

### ACE2005
```bash
python entity_relation_joint_decoder.py \
    --config_file config.yml \
    --save_dir ckpt/ace2005_bert \
    --data_dir data/ACE2005 \
    --fine_tune \
    --device 0
```

### SciERC
```bash
python entity_relation_joint_decoder.py \
    --config_file config.yml \
    --save_dir ckpt/scierc_scibert \
    --data_dir data/SciERC \
    --bert_model_name allenai/scibert_scivocab_uncased \
    --epochs 300 \ 
    --early_stop 50 \
    --fine_tune \
    --device 0
```

Note that a GPU with 32G is required to run the default setting. 
If **OOM** occurs, we suggest that reducing `train_batch_size` and increasing `gradient_accumulation_steps` (`gradient_accumulation_steps` is used to perform *Gradient Accumulation*). 

## Inference
We provide an example ACE2005. 
Note that `save_dir` should contain the trained `best_model`.
```bash
python entity_relation_joint_decoder.py \
    --config_file config.yml \
    --save_dir ckpt/ace2005_bert \
    --data_dir data/ACE2005 \
    --device 0 \
    --log_file test.log \
    --test
```

## Pre-trained Models
We release our pre-trained `UniRE` model for the ACE2005 dataset.

Note that the released model is trained on the `GeForce RTX 2080 Ti` rather than `Quadro RTX 8000`, leading to the performance of the pre-trained model is slightly different from the reported score in the paper.

You can download the BERT based pre-trained model in this [link](https://pan.baidu.com/s/1qXzzYx6Vfgp-YlwRz0yV9A)(password: wc4k) (size: 420M).

Performance of the pre-trained models on the ACE2005 test set:
```
Entity - P: 89.03% R: 88.81% F1: 88.92%
Relation (strict) - P: 68.71% R: 60.25% F1: 64.21%
```

## Cite
If you find our code is useful, please cite:
```
@inproceedings{wang2021unire,
    title = "{UniRE}: A Unified Label Space for Entity Relation Extraction",
    author = "Wang, Yijun and Sun, Changzhi and Wu, Yuanbin and Zhou, Hao and Li, Lei and Yan, Junchi",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```
