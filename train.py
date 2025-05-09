
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from unet_model import simple_unet_model  # Import your U-Net model

# Main Training Script
if __name__ == "__main__":
    # Paths
    file_path = "files/"
    model_path = "files/simple_unet_model"
    os.makedirs(file_path, exist_ok=True)

    # Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 200

    # Assuming train_gen and valid_gen are provided elsewhere
    # Replace these with your actual data generators or arrays
    train_steps = 100  # Placeholder: adjust based on your data
    valid_steps = 20   # Placeholder: adjust based on your data

    # Initialize U-Net model from unet_model.py
    model = simple_unet_model(image_size, image_size, 3, num_classes=1, learning_rate=lr)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), 
                  loss='binary_crossentropy', 
                  metrics=[Recall(), Precision(), MeanIoU(num_classes=2)])

    # Callbacks
    csv_logger = CSVLogger(f"{file_path}unet_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]

    # Train the model
    model.fit(
        train_gen,  # Replace with your training data generator or array
        validation_data=valid_gen,  # Replace with your validation data generator or array
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks
    )
