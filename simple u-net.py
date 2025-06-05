

"""
Simple U-Net model implementation in Keras for image segmentation tasks.
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU

# Default kernel initializer
KERNEL_INITIALIZER = 'he_uniform'


def simple_unet_model(img_height, img_width, img_channels, num_classes=1, learning_rate=1e-3, normalize=False):
    """
    Builds a U-Net model for image segmentation.

    Args:
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
        img_channels (int): Number of channels in the input image (e.g., 3 for RGB).
        num_classes (int, optional): Number of output classes. Defaults to 1 (binary segmentation).
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        normalize (bool, optional): Whether to normalize inputs to [0, 1]. Defaults to False.

    Returns:
        Model: Compiled Keras U-Net model.
    """
    inputs = Input((img_height, img_width, img_channels))

    # Optional input normalization
    if normalize:
        s = Lambda(lambda x: x / 255.0)(inputs)
    else:
        s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(p4)
    c5 = Dropout(0.3)(c5)  # Higher dropout in bottleneck to prevent overfitting
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=KERNEL_INITIALIZER, padding='same')(c9)

    # Output layer: adjust activation based on number of classes
    if num_classes == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
        metrics = [MeanIoU(num_classes=num_classes)]

    outputs = Conv2D(num_classes, (1, 1), activation=activation)(c9)

    # Define and compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    model.summary()

    return model


if __name__ == "__main__":
    # Example usage
    model = simple_unet_model(img_height=256, img_width=256, img_channels=3, num_classes=1)
