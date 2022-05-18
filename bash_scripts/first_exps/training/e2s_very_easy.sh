#!/bin/bash

False=''
True='True'
dataset=e2s_very_easy__train


python -u train.py \
--dataset=$dataset \
--model=resnet50_imagenet \
--mode=simclr_CSI \
--shift_trans_type=rotation \
--batch_size=32 \
--one_class_idx=0 \
