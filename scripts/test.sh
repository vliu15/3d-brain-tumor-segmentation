#!/bin/bash

python test.py \
            --test_folder /share/pi/hackhack/test_test \
            --data_format channels_first \
            --prepro_file /share/pi/hackhack/preprocessed/image_mean_std.npy \
            --chkpt_file ./chkpt.hdf5 \
            --args_file ./config.json \
            --stride 128 \
            --threshold 0.5 \
            --batch_size 8 \
            --gpu
