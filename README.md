Simple U-Net Model for Medical Image Segmentation
Overview
This repository contains unet_model.py, a simple U-Net model for medical image segmentation, designed for skin lesion segmentation on the ISIC 2018 dataset. The U-Net is a convolutional neural network with an encoder-decoder architecture and skip connections, ideal for binary segmentation. Built with TensorFlow 2.17.0 and Python 3.11, it is lightweight and beginner-friendly.
Note: This repository is private. .
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

Prerequisites

Software:
Python 3.11
TensorFlow 2.17.0


Hardware:
GPU recommended (e.g., NVIDIA T4 in Colab).
8GB RAM minimum for inference.


Data (optional):
ISIC 2018 dataset in Google Drive (e.g., /content/drive/MyDrive/ISIC2018/).


Install TensorFlow:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install tensorflow==2.17.0



Usage
Use unet_model.py to create and test the U-Net:
import tensorflow as tf
from unet_model import UNet

# Create model
model = UNet(input_shape=(256, 256, 3), num_filters=64)

# Compile with Dice loss
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=dice_loss, metrics=['accuracy'])

# Model summary
model.summary()

# Test inference
sample_image = tf.random.uniform((1, 256, 256, 3))  # Replace with ISIC 2018 image
prediction = model.predict(sample_image)
print("Prediction shape:", prediction.shape)  # Expected: (1, 256, 256, 1)

Troubleshooting






Shape Errors:

Ensure inputs are (batch, 256, 256, 3):sample_image = tf.random.uniform((1, 256, 256, 3))
print(model(sample_image).shape)  # Expected: (1, 256, 256, 1)




TensorFlow Version:

Check:pip show tensorflow

Use tensorflow==2.17.0.



Contributing
To contribute:

Request access to the private repository.
Fork (if granted access).
Create a branch: git checkout -b feature/YourFeature.
Commit: git commit -m 'Add YourFeature'.
Push: git push origin feature/YourFeature.
Open a Pull Request.

License
MIT License. See LICENSE.
Acknowledgments

ISIC 2018: Dataset by International Skin Imaging Collaboration.
TensorFlow: Framework for model implementation.
U-Net: Architecture by Ronneberger et al. (2015).

Contact


Last updated: April 25, 2025
