#!/bin/bash

False=''
True='True'
dataset=cars3d__az_id__train


python -u -m torch.distributed.launch --nproc_per_node=4 train.py \
--dataset=$dataset \
--model=resnet50 \
--mode=simclr_CSI \
--shift_trans_type=rotation \
--batch_size=32 \
--one_class_idx=0 \



