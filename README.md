Simple U-Net Model for Medical Image Segmentation

Overview
This repository contains the implementation of a simple U-Net model (unet_model.py) for medical image segmentation, designed for skin lesion segmentation on the ISIC 2018 dataset. U-Net is a convolutional neural network with an encoder-decoder architecture and skip connections to preserve spatial details, making it effective for binary segmentation tasks. This lightweight, beginner-friendly implementation uses TensorFlow 2.17.0 and Python 3.11.
Note: This repository is private. Contact your-email@example.com for access.
Table of Contents

Model Architecture
Features
Prerequisites
Installation
Usage
Troubleshooting
Contributing
License
Acknowledgments
Contact

Model Architecture
The U-Net model consists of:

Encoder: Four downsampling blocks, each with two 3x3 convolutions (ReLU activation, same padding), followed by 2x2 max-pooling.

Bottleneck: Two 3x3 convolutions with increased filters.

Decoder: Four upsampling blocks, each with a 2x2 transposed convolution, concatenation with corresponding encoder features (skip connections), and two 3x3 convolutions.

Output: 1x1 convolution with sigmoid activation for binary segmentation.

Input Shape: (256, 256, 3) (RGB images).

Output Shape: (256, 256, 1) (binary segmentation mask).

Filters: Starts at 64, doubling per encoder block (64, 128, 256, 512, 1024).


Features

Encoder-Decoder Structure: Captures contextual and spatial information.
Skip Connections: Preserves fine-grained details via feature concatenation.
Lightweight Design: Minimal layers for efficient training on Google Colab.
Binary Segmentation: Outputs probability maps for skin lesions (0 or 1).

Prerequisites

Software:
Python 3.11
TensorFlow 2.17.0


Hardware:
GPU recommended (e.g., NVIDIA T4 in Google Colab).
Minimum 8GB RAM for inference.


Data (optional, for usage):
ISIC 2018 dataset (ISIC2018_Task1-2_Training_Input and ISIC2018_Task1_Training_GroundTruth).
Store in Google Drive (e.g., /content/drive/MyDrive/ISIC2018/).



Installation

Clone the Repository:

Requires access to the private repository.git clone git@github.com:your-username/simple-unet-model.git
cd simple-unet-model





