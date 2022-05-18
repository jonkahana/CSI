#!/bin/bash

False=''
True='True'
dataset=edges2shoes_x64__domain_vs_shoe_type_short__train


python -u train.py \
--dataset=$dataset \
--model=resnet18_imagenet \
--img-size=224 \
--mode=simclr_CSI \
--shift_trans_type=rotation \
--batch_size=32 \
--one_class_idx=0 \


python -u eval_Red_PANDA.py \
--mode ood_pre \
--dataset=$dataset \
--model=resnet18_imagenet \
--img-size=224 \
--ood_score=CSI \
--shift_trans_type=rotation \
--print_score \
--ood_samples=10 \
--resize_factor=0.54 \
--resize_fix \


