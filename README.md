Simple U-Net Model for Medical Image Segmentation

Overview

This repository contains unet_model.py, a simple U-Net model for medical image segmentation, designed for skin lesion segmentation on the ISIC 2018 dataset. The U-Net is a convolutional neural network with an encoder-decoder architecture and skip connections, ideal for binary segmentation. Built with TensorFlow 2.17.0 and Python 3.11, it is lightweight and beginner-friendly.

Note: This repository is private. Contact your-email@example.com for access.

Model Architecture

The U-Net model includes:





Encoder: Four blocks, each with two 3x3 convolutions (ReLU, same padding) and 2x2 max-pooling.



Bottleneck: Two 3x3 convolutions with increased filters.



Decoder: Four blocks, each with a 2x2 transposed convolution, concatenation with encoder features (skip connections), and two 3x3 convolutions.



Output: 1x1 convolution with sigmoid activation for binary segmentation.



Input: (256, 256, 3) (RGB images).



Output: (256, 256, 1) (binary mask).



Filters: Starts at 64, doubles per encoder block (64, 128, 256, 512, 1024).

Features





Encoder-decoder captures context and spatial details.



Skip connections preserve fine-grained features.



Lightweight for Google Colab training.



Outputs binary segmentation masks for skin lesions.

