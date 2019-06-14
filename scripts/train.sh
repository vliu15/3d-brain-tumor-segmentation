#!/bin/bash

python train.py \
    --train_loc /share/pi/hackhack/preprocessed_v2/train.image_wise.tfrecords \
    --val_loc /share/pi/hackhack/preprocessed_v2/val.image_wise.tfrecords \
    --prepro_loc /share/pi/hackhack/preprocessed_v2/image_mean_std.npy \
    --data_format channels_first \
    --log_file train.log \
    --save_file chkpt.hdf5 \
    --gpu
