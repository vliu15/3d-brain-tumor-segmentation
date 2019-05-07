# Volumetric Brain Tumor Segmentation
We seek to replicate the model that won the BraTS 2018 Segmentation Challenge. The top model was created by the NDVLMED team under Andriy Myromenko, who is the first author of the [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf) paper that we use for reference.


## TODO:
 - [x] Convolutional encoder
 - [x] Convolutional decoder
 - [ ] Variational autoencoder
 - [x] Loss function: soft dice + KL divergence + L2
 - [ ] Adam optimizer with scheduled learning rate
 - [ ] Preprocessing
 - [ ] Training / validation cycles
 