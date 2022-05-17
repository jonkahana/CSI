#!/bin/bash

False=''
True='True'
dataset=cars3d__az_id__train


python -u train.py \
--dataset=$dataset \
--model=resnet18_imagenet \
--mode=simclr_CSI \
--shift_trans_type=rotation \
--batch_size=32 \
--one_class_idx=0 \
