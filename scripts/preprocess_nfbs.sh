#!/usr/bin/python

python preprocess.py \
    --in_locs /share/pi/hackhack/vincent/NFBS_Dataset \
    --modalities T1w. \
    --truth T1w_brainmask \
    --out_loc /share/pi/hackhack/vincent/nfbs \
    --create_val
