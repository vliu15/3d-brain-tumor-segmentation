# Volumetric Brain Tumor Segmentation
We seek to replicate the model that won the BraTS 2018 Segmentation Challenge. The top model was created by the NVDLMED team under Andriy Myronenko, who is the first author of the [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf) paper that we use for reference.


## Usage
Dependencies are only supported for Python3 and can be found in `requirements.txt`. We use `numpy==1.15` for the bulk of preprocessing and `tensorflow==2.0.0-alpha0` for the model architecture, utilizing `tf.keras.Model` and `tf.keras.Layer` subclassing.

### BraTS Data
The BraTS 2017/2018 dataset is not publicly available, so we cannot provide download scripts for those. Once downloaded, run preprocessing on the original data format, which should look something like this:
```
BraTS17TrainingData/*/*/*[t1,t1ce,t2,flair,seg].nii.gz
```

### Preprocessing
For each example, there are 4 modalities and 1 label, each of shape `240 x 240 x 155`. As per the paper, we concatenate the modalities together and perform the following preprocessing steps:
 - Compute per-channel voxel-wise `mean` and `std` and normalize per channel *for the training set*.
 - Shift each channel by `[-0.1, 0.1] * std` then scale each channel by `[0.9, 1.1]`.
 - With probability `0.5`, flip and add each image to the dataset.
 - Randomly crop a `160 x 192 x 128` subsection from each image.
 - Serialize to `tf.TFRecord` format for portability.

*Note that the preprocessed data is memory-intensive.* The training and validation `.tfrecords` files are around 31GB and 2GB, respectively. We also shard the data in run-time (shards of around size 12) so that the corresponding Numpy operations can fit on CPU. To run preprocessing,
```
python3 preprocess.py --brats_folder data/BraTS17TrainingData --create_val
```

> All command-line arguments can be found in `utils/arg_parser.py`.

> There are 285 training examples in the BraTS 2017/2018 training sets, but since we do not have access to the validation set, we opt for a 10:1 split and end up with 260 and 25 training and validation examples, respectively. To create this split, run with the `--create_val` flag.

### Training
As per the paper, we replicate all hyperparameters used in training (as default arguments). We will provide our training logs and graphs here shortly. To run training,
```
python3 train.py --train_loc data/train.tfrecords --val_loc data/val.tfrecords
```

> Use the `--gpu` flag to run on GPU.

## TODO:
 - [x] Convolutional encoder
 - [x] Convolutional decoder
 - [x] Variational autoencoder
 - [x] Loss function: soft dice + KL divergence + L2
 - [x] Adam optimizer with scheduled learning rate
 - [x] Preprocessing
 - [x] Training / validation cycles
 - [ ] Checkpointing: saving and restoring
 - [ ] Evaluation metrics: dice, sensitivity, specificity, Hausdorff distances
 - [ ] Train on GPU
