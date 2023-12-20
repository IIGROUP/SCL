# Seeing What You Miss: Vision-Language Pre-training with Semantic Completion Learning

This is a PyTorch/GPU implementation of the paper [SCL](https://arxiv.org/pdf/2211.13437.pdf). 


## Install

Our environment: CUDA11.3, torch1.11.0, torchvision0.12.0.

```bash
pip install -r requirements.txt
```

## Weights and ALIGN-BENCH

Our pre-trained and fine-tuned model weights can be downloaded at [huggingface/SCL](https://huggingface.co/jiyatai/SCL).

Our developed ALIGN-BENCH can be gained at [huggingface/ALIGN-BENCH](https://huggingface.co/datasets/jiyatai/ALIGN-BENCH).

## Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Pre-training

```bash
python run.py --task pretrain
```

The detailed settings can be found in './scl/config.py', like pretraining datasets, optimation arguments, input size. 

Note that 'plugins=[MyCluster(), MyDDPPlugin()]' of pl.Trainer(run.py) is used in multi-nodes ddp training.

## Downstream Tasks

```bash
python run.py --task vqa/nlvr2/f30k/coco/msrvtt/lsmdc
```


## Acknowledgements

The code is based on [METER](https://github.com/zdou0830/METER) and [VLC](https://github.com/guilk/VLC).

