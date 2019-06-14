#!/bin/bash

python train.py \
    --train_loc /share/pi/hackhack/preprocessed_v2/train.image_wise.tfrecords \
    --val_loc /share/pi/hackhack/preprocessed_v2/val.image_wise.tfrecords \
    --prepro_loc /share/pi/hackhack/preprocessed_v2/image_mean_std.npy \
    --data_format channels_first \
    --log_file train.log \
    --save_file chkpt.hdf5 \
    --gn_groups 8 \
    --se_reduction 4 \
    --base_filters 32 \
    --depth 4 \
    --l2_scale 1e-5 \
    --n_epochs 150 \
    --downsamp conv \
    --upsamp conv \
    --gpu
