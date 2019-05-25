"""Contains frequently used constants."""
# Preprocessing.
# BRATS_MODALITIES = ['t1', 't1ce', 'flair', 't2']
RTOG_MODALITIES = ['t1c', 'flair']
BRATS_MODALITIES = ['t1ce', 'flair']
TRUTH = 'seg'

# LABELS      =       [1, 2, 4]
LABELS      =       [0, 1, 2, 3]

# Data.
RAW_H    =  240                     # Height of raw image.
RAW_W    =  240                     # Width of raw image.
RAW_D    =  155                     # Depth of raw image.
H        =  128                     # Target height of preprocessed image.
W        =  128                     # Target width of preprocessed image.
D        =  128                     # Target depth of preprocessed image.

IN_CH    =  len(BRATS_MODALITIES)   # Number of input channels.
OUT_CH   =  len(LABELS)             # Number of output channels.

CHANNELS_FIRST_X_SHAPE     =   (IN_CH, H, W, D)
CHANNELS_FIRST_Y_SHAPE     =   (1, H, W, D)
CHANNELS_LAST_X_SHAPE      =   (H, W, D, IN_CH)
CHANNELS_LAST_Y_SHAPE      =   (H, W, D, 1)


# Encoder.
ENC_CONV_LAYER_SIZE         =       32

ENC_CONV_BLOCK0_SIZE        =       32
ENC_CONV_BLOCK0_NUM         =       2

ENC_CONV_BLOCK1_SIZE        =       64
ENC_CONV_BLOCK1_NUM         =       2

ENC_CONV_BLOCK2_SIZE        =       128
ENC_CONV_BLOCK2_NUM         =       2

ENC_CONV_BLOCK3_SIZE        =       256
ENC_CONV_BLOCK3_NUM         =       4


# Decoder.
DEC_CONV_BLOCK2_SIZE        =       128
DEC_CONV_BLOCK1_SIZE        =       64
DEC_CONV_BLOCK0_SIZE        =       32

DEC_CONV_LAYER_SIZE         =       32


# Variational Autoencoder.
VAE_VD_CONV_SIZE            =       16
VAE_VD_BLOCK_SIZE           =       256

VAE_LATENT_SIZE             =       128

VAE_VU_BLOCK_SIZE           =       256

VAE_CONV_BLOCK2_SIZE        =       128
VAE_CONV_BLOCK1_SIZE        =       64
VAE_CONV_BLOCK0_SIZE        =       32

# Loss.
GDL_WEIGHT  =       1.0
SS_WEIGHT   =       1.0
L2_WEIGHT   =       0.1
KL_WEIGHT   =       0.1
FL_WEIGHT   =       1.0
