"""Contains frequently used constants."""
# Preprocessing.
ALL_MODALITIES = ['t1', 't1ce', 'flair', 't2']
TRUTH = 'seg'


# Data.
CHANNELS_FIRST_X_SHAPE     =   (-1, 4, 160, 192, 128)
CHANNELS_FIRST_Y_SHAPE     =   (-1, 1, 160, 192, 128)
CHANNELS_LAST_X_SHAPE      =   (-1, 160, 192, 128, 4)
CHANNELS_LAST_Y_SHAPE      =   (-1, 160, 192, 128, 1)

RAW_H    =  240                     # Height of raw image.
RAW_W    =  240                     # Width of raw image.
RAW_D    =  155                     # Depth of raw image.
H        =  160                     # Target height of preprocessed image.
W        =  192                     # Target width of preprocessed image.
D        =  128                     # Target depth of preprocessed image.

IN_CH    =  len(ALL_MODALITIES)     # Number of input channels.
OUT_CH   =  3                       # Number of output channels.


# Encoder.
ENC_CONV_LAYER_SIZE         =       32      # Number of filters of initial conv layer.

ENC_CONV_BLOCK0_SIZE        =       32      # Number of filters of first conv block.
ENC_CONV_BLOCK0_NUM         =       2       # Number of first conv blocks.

ENC_CONV_BLOCK1_SIZE        =       64      # Number of filters of second conv block.
ENC_CONV_BLOCK1_NUM         =       2       # Number of second conv blocks.

ENC_CONV_BLOCK2_SIZE        =       128     # Number of filters of third conv block.
ENC_CONV_BLOCK2_NUM         =       2       # Number of third conv blocks.

ENC_CONV_BLOCK3_SIZE        =       256     # Number of filters of fourth conv block.
ENC_CONV_BLOCK3_NUM         =       4       # Number of fourth conv blocks.


# Decoder.
DEC_CONV_BLOCK2_SIZE        =       128     # Number of filters in first conv block.
DEC_CONV_BLOCK1_SIZE        =       64      # Number of filters in second conv block.
DEC_CONV_BLOCK0_SIZE        =       32      # Number of filters in third conv block.

DEC_CONV_LAYER_SIZE         =       32      # Number of filters in final conv layer.


# Variational Autoencoder.
VAE_VD_CONV_SIZE            =       16      # Number of filters in VD conv layer.
VAE_VD_BLOCK_SIZE           =       256     # Dimensionality of flattened size.

VAE_VU_BLOCK_SIZE           =       256     # Number of filters in VU conv block.

VAE_CONV_BLOCK2_SIZE        =       128     # Number of filters in second conv block.
VAE_CONV_BLOCK1_SIZE        =       64      # Number of filters in first conv block.
VAE_CONV_BLOCK0_SIZE        =       32      # Number of filters in zeroth conv block.