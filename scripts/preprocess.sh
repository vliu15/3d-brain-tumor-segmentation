#!/bin/bash

python preprocess.py \
            --brats_folder /share/pi/hackhack/RTOG_341samples/BRATS2017/brats2017 \
            --out_folder /share/pi/hackhack/preprocessed \
            --data_format channels_last \
            --create_val \
            --norm image \
            --mirror_prob 0.75 \
            --n_crops 3 \
