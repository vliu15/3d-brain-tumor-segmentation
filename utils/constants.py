"""Contains frequently used constants."""

ALL_MODALITIES = ['t1', 't1ce', 'flair', 't2']
TRUTH = 'seg'

CHANNELS_FIRST_X_SHAPE     =   (-1, 4, 160, 192, 128)
CHANNELS_FIRST_Y_SHAPE     =   (-1, 1, 160, 192, 128)
CHANNELS_LAST_X_SHAPE      =   (-1, 160, 192, 128, 4)
CHANNELS_LAST_Y_SHAPE      =   (-1, 160, 192, 128, 1)

RAW_H    =  240
RAW_W    =  240
RAW_D    =  155
H        =  160
W        =  192
D        =  128
IN_CH    =  4
OUT_CH   =  3
