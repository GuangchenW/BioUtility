# Training a SAM2UNet Model in Google Colab Environment

This folder contains the scripts and helpers for training a UNet segmentation model based on [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet/tree/main).

`frame_extractor.py`: For extracting a single frame from a z-stacked tif image, with optional visual enhancements for facilitating manual annotation.

`data_augmentation.py`: Perform data augmentation on a set of images and corresponding labels (mask annotations). Generates an expanded dataset based on possible variations due to acquistion.

`sam2unet.ipynb`: Main training/test script.

See comments in scripts for detailed usage instructions.