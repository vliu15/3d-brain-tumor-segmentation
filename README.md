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

The model can be found in `model/model.py` and contains an `inference` mode in addition to the `training` mode that `tf.Keras.Model` supports.
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

```
python preprocess.py \
    --in_locs /path/to/BraTS17TrainingData \
    --modalities t1ce,flair \
    --truth seg \
    --create_val
```

> All command-line arguments can be found in `args.py`.

> There are 285 training examples in the BraTS 2017/2018 training sets, but for lack of validation set, the `--create_val` flag creates a 10:1 split, resulting in 260 and 25 training and validation examples, respectively.

### Training
Most hyperparameters proposed in the paper are used in training. The input is randomly flipped across spatial axes with probability 0.5 and cropped to `128 x 128 x 128` per example in training (making the training data stochastic). The validation set is dynamically created each epoch in a similar fashion.
```
python train.py \
    --train_loc /path/to/train \
    --val_loc /path/to/val \
    --prepro_file /path/to/prepro/prepro.npy \
    --save_folder checkpoint \
    --crop_size 128,128,128
```

> Use the `--gpu` flag to run on GPU.

### Testing: Generating Segmentation Masks
The testing script `test.py` runs inference on unlabeled data provided as input by generating sample labels on the whole image, padded to a size that is compatible with downsampling. The VAE is not run in inference so the model is actually fully convolutional.
```
python test.py \
    --in_locs /path/to/test \
    --modalities t1ce,flair \
    --prepro_loc /path/to/prepro/prepro.npy \
    --tumor_model checkpoint
```
*Training arguments are saved in the checkpoint folder. This bypasses the need for manual model initialization.*

> The `Interpolator` class is used to interpolate voxel sizes in rescaling so that all inputs can be resized to 1 mm^3.

> NOTE: `test.py` is not fully debugged and functional. If needed please open an issue.


### Skull Stripping
Because BraTS contains skull-stripped images which are uncommon in actual applications, we support training and inference of skull stripping models. The same pipeline can be generalized, but using the NFBS skull-stripping dataset [here](http://preprocessed-connectomes-project.org/NFB_skullstripped/). Note that in model initialization and training, the number of output channels `--out_ch` would be different for these tasks.

> If the testing data contains skull bits, run skull stripping and tumor segmentation sequentially in inference time by specifying the `--skull_model` flag. All preprocessing and training should work for both tasks as is.

### Results
We run training on a V100 32GB GPU with a batch size of 1. Each epoch takes around ~12 minutes to run. Below is a sample training curve, using all default model parameters.

|Epoch|Training Loss|Training Dice Score|Validation Loss|Validation Dice Score|
|:---:|:-----------:|:-----------------:|:-------------:|:-------------------:|
|0    |1.000        |0.134              |0.732          |0.248                |
|50   |0.433        |0.598              |0.413          |0.580                |
|100  |0.386        |0.651              |0.421          |0.575                |
|150  |0.356        |0.676              |0.393          |0.594                |
|200  |0.324        |0.692              |0.349          |0.642                |
|250  |0.295        |0.716              |0.361          |0.630                |
|300  |0.282        |0.729              |0.352          |0.644                |
