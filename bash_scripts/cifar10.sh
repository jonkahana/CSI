#!/bin/bash

False=''
True='True'
dataset=cars3d__az_id__train


python -u train.py \
--dataset=cifar10 \
--model=resnet18 \
--mode=simclr_CSI \
--shift_trans_type=rotation \
--batch_size=32 \
--original-datasets=$True \
--one_class_idx=0 \
