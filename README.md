# Volumetric Brain Tumor Segmentation
We seek to build off of the model that won the BraTS 2018 Segmentation Challenge. The top model was created by the NVDLMED team under Andriy Myronenko, who is the first author of the [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf) paper that we use for reference.

## Model
We adopt the encoder-decoder convolutional architecture described in the [above paper](https://arxiv.org/pdf/1810.11654.pdf) with a variational autoencoder branch for regularizing the encoder. There are several architectural changes that we have made based on common practices combined with the incompleteness of the paper in some areas. We have
 - Added squeeze-excitation layers (SENet) in the ResNet blocks, as they have been shown to improve performance.
 - Reordered all convolutional layers to consist of `Conv3D`, `GroupNorm`, `ReLU`, except for all pointwise and output layers.
 - Replaced strided convolutions in downsampling with max pooling.
 - Use He normal initialization for *all* layer kernels except those with sigmoid activations, which we use Glorot normal initialization for.
 - Add per-epoch learning rate linear warmup from a base learning rate of `1e-6` for the first 10 epochs.

> Use the `--norm [group, batch, layer]` flag to specify the normalization mechanism. Defaults to `group` for group normalization.

> Use the `--downsamp [max, avg, conv]` flag to specify the downsampling method. Defaults to `max` for max pooling.

> Use the `--upsamp [linear, conv]` flag to specify the upsampling method. Defaults to `linear` for linear upsampling.

## Usage
Dependencies are only supported for Python3 and can be found in `requirements.txt`. We use `numpy==1.15` for the bulk of preprocessing and `tensorflow==2.0.0-alpha0` for the model architecture, utilizing `tf.keras.Model` and `tf.keras.Layer` subclassing.

### BraTS Data
The BraTS 2017/2018 dataset is not publicly available, so we cannot provide download scripts for those. Once downloaded, run preprocessing on the original data format, which should look something like this:
```
BraTS17TrainingData/*/*/*[t1,t1ce,t2,flair,seg].nii.gz
```

### Preprocessing
For each example, there are 4 modalities and 1 label, each of shape `240 x 240 x 155`. We slightly adjust the preprocessing steps described in the paper to match our use case with Stanford Medicine:
 - Concatenate the `t1ce` and `flair` modalities along the channel dimension.
 - Compute per-channel image-wise `mean` and `std` and normalize per channel *for the training set*.
 - Serialize to `tf.TFRecord` format for convenience in training.

The training and validation `.tfrecords` files are around 30GB and 3GB, respectively. See `scripts/preprocess.sh` for a detailed example of how to run preprocessing.
```
python preprocess.py --brats_folder data/BraTS17TrainingData --create_val
```

> All command-line arguments can be found in `utils/arg_parser.py`.

> There are 285 training examples in the BraTS 2017/2018 training sets, but since we do not have access to the validation set, we opt for a 10:1 split and end up with 260 and 25 training and validation examples, respectively. To create this split, run with the `--create_val` flag.

### Training
As per the paper, we adopt all hyperparameters used in training (as default arguments). We will provide our training logs and graphs here shortly. See `scripts/train.sh` for a detailed example of how to run training. The size of the model can be adjusted in `utils/constants.py`. We randomly flip across spatial axes with probability 0.5 and sample 1 crop per example in training (essentially making the training data stochastic). The validation set is created at the beginning of training and remains static.
```
python train.py --train_loc data/train.image_wise.tfrecords --val_loc data/val.image_wise.tfrecords
```

> Use the `--n_val_sets` flag to specify how many crops to take to create the validation set. Increase for robustness.

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
We run training on a V100 32GB GPU. Each epoch takes around ~10-15 minutes to run. Below is a sample training curve, using all default model parameters.

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
