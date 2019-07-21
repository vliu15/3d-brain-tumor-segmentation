#!/bin/bash

python preprocess.py \
            --in_locs /share/pi/hackhack/RTOG_341samples/BRATS2017/brats2017/train,/share/pi/hackhack/RTOG_341samples/BRATS2017/brats2017/val \
            --modalities t1ce,flair \
            --truth seg \
            --out_loc /share/pi/hackhack/vincent/preprocessed_v2 \
            --create_val
