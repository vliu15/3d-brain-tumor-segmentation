#!/bin/bash

python train.py \
    --train_loc /share/pi/hackhack/preprocessed/train.image_wise.tfrecords \
    --val_loc /share/pi/hackhack/preprocessed/val.image_wise.tfrecords \
    --prepro_loc /share/pi/hackhack/preprocessed/image_mean_std.npy \
    --args_file config.json \
    --log_file train.log \
    --save_file chkpt.hdf5 \
    --gpu
