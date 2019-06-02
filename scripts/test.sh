#!/bin/bash

python test.py \
            --test_folder /share/pi/hackhack/RTOG_341samples/rtog_nii_n342/post \
            --data_format channels_last \
            --chkpt_file chkpt.hdf5 \
            --prepro_file /share/pi/hackhack/preprocessed/image_mean_std.npy \
            --use_se \
            --gpu