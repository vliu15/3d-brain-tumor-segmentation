#!/bin/bash

python preprocess.py \
            --brats_folder /share/pi/hackhack/RTOG_341samples/BRATS2017/brats2017 \
            --out_folder /share/pi/hackhack/preprocessed \
            --create_val \
            --norm image
