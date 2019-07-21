#!/bin/bash

python train.py \
    --train_loc /share/pi/hackhack/vincent/preprocessed_v2/train \
    --val_loc /share/pi/hackhack/vincent/preprocessed_v2/val \
    --prepro_loc /share/pi/hackhack/vincent/preprocessed_v2/prepro.npy \
    --save_folder checkpoint \
    --data_format channels_first \
    --gpu
