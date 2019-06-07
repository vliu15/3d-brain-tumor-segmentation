#!/bin/bash

python test.py \
            --test_folder /share/pi/hackhack/test_viz \
            --data_format channels_first \
            --chkpt_file chkpt.hdf5 \
            --prepro_file /share/pi/hackhack/preprocessed/image_mean_std.npy \
            --gpu