#!/bin/bash

python train.py \
    --train_loc /share/pi/hackhack/preprocessed/train.image_wise.tfrecords \
    --val_loc /share/pi/hackhack/preprocessed/val.image_wise.tfrecords \
    --data_format data_format \
    --log_file train.log \
    --save_file chkpt.hdf5 \
    --log_steps -1 \
    --patience 10 \
    --n_epochs 300 \
    --lr 1e-5 \
    --warmup_epochs 10 \
    --batch_size 1 \
    --use_se \
    --gpu
