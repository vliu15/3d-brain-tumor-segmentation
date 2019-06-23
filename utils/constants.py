"""Contains frequently used constants."""

RTOG_MODALITIES     = ['t1c', 'flair']
BRATS_MODALITIES    = ['t1ce', 'flair']
TRUTH               = 'seg'
LABELS              = [1, 2, 4]

RAW_H    =  240                     # Height of raw image.
RAW_W    =  240                     # Width of raw image.
RAW_D    =  155                     # Depth of raw image.
H        =  128
W        =  128
D        =  128

IN_CH    =  len(BRATS_MODALITIES)   # Number of input channels.
OUT_CH   =  len(LABELS)             # Number of output channels.
