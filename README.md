# Volumetric Brain Tumor Segmentation
This repository experiments with best techniques to improve dense, volumetric semantic segmentation. Specifically, the model is of U-net architectural style and includes variational autoencoder (for regularization), residual blocks, spatial and channel squeeze-excitation layers, and dense connections.

## Model
This is a variation of the [U-net](https://arxiv.org/pdf/1606.06650.pdf) architecture with [variational autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf). There are several architectural enhancements, including
 - [Spatial and channel squeeze-excitation layers](https://arxiv.org/abs/1803.02579) in the ResNet blocks.
 - [Dense connections](https://arxiv.org/pdf/1608.06993.pdf) between encoder ResNet blocks at the same spatial resolution level.
 - Convolutional layers to consist of order `[Conv3D, GroupNorm, ReLU]`, except for all pointwise and output layers.
 - He normal initialization for *all* layer kernels except those with sigmoid activations, which are initialized with Glorot normal.
 - Convolutional downsampling and upsampling operations.

## Usage
Dependencies are only supported for Python3 and can be found in `requirements.txt` (`numpy==1.15` for preprocessing and `tensorflow==2.0.0-alpha0` for model architecture, utilizing `tf.keras.Model` and `tf.keras.Layer` subclassing).

The `VolumetricCNN` model can be found in `model/model.py` and contains an `inference` mode in addition to the `training` mode that `tf.Keras.Model` supports.
 - Specify `training=False, inference=True` to only receive the decoder output, as desired in test time.
 - Specify `training=False, inference=False` to receive both the decoder and variational autoencoder output to be able to run loss and metrics, as desired in validation time.

### BraTS Data
The BraTS 2017/2018 dataset is not publicly available, so download scripts for those are not available. Once downloaded, run preprocessing on the original data format, which should look something like this:
```
BraTS17TrainingData/*/*/*[t1,t1ce,t2,flair,seg].nii.gz
```

### Preprocessing
For each example, there are 4 modalities and 1 label, each of shape `240 x 240 x 155`. Preprocessing steps consist of:
 - Concatenate the `t1ce` and `flair` modalities along the channel dimension.
 - Compute per-channel image-wise `mean` and `std` and normalize per channel *for the training set*.
 - Crop as much background as possible across all images. Final image sizes are `155 x 190 x 147`.
 - Serialize to `tf.TFRecord` format for convenience in training.

The training and validation `.tfrecords` files are around 30GB and 3GB, respectively. See `scripts/preprocess.sh` for a detailed example of how to run preprocessing.
```
python preprocess.py --brats_folder data/BraTS17TrainingData --create_val
```

> All command-line arguments can be found in `utils/arg_parser.py`.

> There are 285 training examples in the BraTS 2017/2018 training sets, but for lack of validation set, the `--create_val` flag creates a 10:1 split, resulting in 260 and 25 training and validation examples, respectively.

### Training
All hyperparameters proposed in the paper are used in training. See `scripts/train.sh` for a detailed example of how to run training. The size of the model can be adjusted in `utils/constants.py`. The input is randomly flipped across spatial axes with probability 0.5 and cropped to `128 x 128 x 128` per example in training (essentially making the training data stochastic). The validation set is created at the beginning of training and remains static.
```
python train.py --train_loc data/train.image_wise.tfrecords --val_loc data/val.image_wise.tfrecords --prepro_file data/image_mean_std.npy --args_file config.json
```

> Use the `--n_val_sets` flag to specify how many crops to take to create the validation set. Increase for metric robustness.

> Use the `--gpu` flag to run on GPU.

### Testing: Generating Segmentation Masks
The testing script `test.py` run inference on unlabeled data provided as input by generating sample labels on each crop of the image, then stitching them together to form the complete mask over the whole image. See `scripts/test.sh` for a detailed example of how to run testing.
```
python test.py --test_folder /path/to/test/data --prepro_file data/image_mean_std.npy --chkpt_file chkpt.hdf5
```
*You must specify the same model parameters as the ones use in training for the trained weights to be successfully loaded.*

> Use the `--stride` to indicate the stride between cropping to cover the input. `64` is used as default, and each image takes around ~20 seconds.

> Use the `--batch_size` flag to indicate the number of crops fed into the model at a time, per example.

> The `Interpolator` class is used to interpolate linearly depthwise, if some of the inputs are shallower in depth.

### Results
We run training on a V100 32GB GPU. Each epoch takes around ~10 minutes to run. Below is a sample training curve, using all default model parameters.

|Epoch|Training Loss|Training Dice Score|Validation Loss|Validation Dice Score|
|:---:|:-----------:|:-----------------:|:-------------:|:-------------------:|
|0    |127.561      |0.012              |121.369        |0.020                |
|10   |42.413       |0.508              |38.686         |0.553                |
|20   |37.583       |0.504              |36.580         |0.582                |
|30   |34.323       |0.490              |33.333         |0.539                |
|40   |31.133       |0.505              |30.656         |0.589                |
|50   |28.587       |0.539              |30.266         |0.594                |
|60   |26.653       |0.593              |27.696         |0.604                |
|70   |24.800       |0.607              |27.082         |0.624                |

## TODO:
 - [ ] Add preprocessing/training/inference on skull-stripping data for test cases.
