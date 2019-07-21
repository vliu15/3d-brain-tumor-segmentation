#!/usr/bin/python

python train.py \
    --train_loc /share/pi/hackhack/vincent/nfbs/train.tfrecords \
    --val_loc /share/pi/hackhack/vincent/nfbs/val.tfrecords \
    --prepro_loc /share/pi/hackhack/vincent/nfbs/prepro.npy \
    --save_folder checkpoint_nfbs \
    --out_ch 1 \
    --gpu
