#!/bin/bash

False=''
True='True'
dataset=edges2shoes_x64__domain_vs_shoe_type_short__train


python -u train.py \
--dataset=$dataset \
--model=resnet18_imagenet \
--mode=simclr_CSI \
--shift_trans_type=rotation \
--batch_size=32 \
--one_class_idx=0 \
