# Training a SAM2UNet Model in Google Colab Environment

This folder contains the script and helpers for training a UNet segmentation model based on [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet/tree/main).

`frame_extractor.py`: For extracting a single frame from a z-stacked tif image, with optional visual enhancements for facilitating manual annotation.

`data_augmentation.py`: Perform data augmentation on a set of images and corresponding labels (mask annotations). Generates an expanded dataset based on possible variations due to acquistion.

`sam2unet.ipynb`: Main training/test script.

## Make annotations

Images can be manually annotated using Computer Vision Annoataion Tool (CVAT), a free online platform for annotating images.

CVAT can be accessed [here](https://www.cvat.ai/). While the free version has limited functionalities, it is sufficient for small scale manual annotations.

## Model training

Once masks are obtains, they may be augmented using `data_augmentation.py`. For more on why augmenting data, see [Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html).

To run `data_augmentation.py`, please first install the dependencies via `pip install -r requirements.txt`. For how to run the script, use `python data_augmentation.py -h`.

After data augmentation, you should have the augmented images and masks. These will be used for model training.

To use `sam2unet.ipynb`, run it in a Google Colab environment. Alternatively, run it in a local Jupyter Notebook environment. However, at least 16GB of GPU VRAM is recommended.

See comments in the notebook for detailed usage instructions.
