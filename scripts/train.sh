#!/bin/bash

python train.py \
    --train_loc /share/pi/hackhack/preprocessed/train.image_wise.tfrecords \
    --val_loc /share/pi/hackhack/preprocessed/val.image_wise.tfrecords \
    --data_format channels_first \
    --log_file train-8.log \
    --save_file chkpt-8.hdf5 \
    --log_steps -1 \
    --patience 10 \
    --n_epochs 300 \
    --lr 1e-5 \
    --warmup_epochs 10 \
    --batch_size 1 \
    --decoder_loss dice \
    --n_val_sets 2 \
    --gpu
