import os
import zipfile
import gdown
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import nibabel as nib
from model import unet_model
import matplotlib.pyplot as plt

def download_dataset():
    """Download and extract BraTS dataset"""
    print("Downloading dataset...")
    url = "https://drive.google.com/uc?id=1A2IU8Sgea1h3fxlpkvJbgM1VrG4X0rT7"
    output = "BraTS2020_TrainingData.zip"
    
    # Create directory if needed
    if not os.path.exists("BraTS2020_TrainingData"):
        os.makedirs("BraTS2020_TrainingData")
    
    # Download file
    gdown.download(url, output, quiet=False)
    
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("BraTS2020_TrainingData")
    print("Dataset ready")

def load_nii_file(file_path):
    """Load NIfTI file and return as numpy array"""
    img = nib.load(file_path)
    return img.get_fdata()

def preprocess_data():
    """Preprocess MRI images and masks"""
    images = []
    masks = []
    
    patients = [d for d in os.listdir("BraTS2020_TrainingData") if os.path.isdir(os.path.join("BraTS2020_TrainingData", d))]
    
    for patient in patients:
        patient_path = os.path.join("BraTS2020_TrainingData", patient)
        
        # Load flair image (channel 0)
        flair_path = os.path.join(patient_path, f"{patient}_flair.nii")
        flair_img = load_nii_file(flair_path)
        
        # Load segmentation mask
        seg_path = os.path.join(patient_path, f"{patient}_seg.nii")
        seg_img = load_nii_file(seg_path)
        
        # For simplicity, use middle slice
        mid_slice = flair_img.shape[2] // 2
        image_slice = flair_img[:, :, mid_slice]
        mask_slice = seg_img[:, :, mid_slice]
        
        # Threshold mask to binary (tumor vs non-tumor)
        mask_slice = (mask_slice > 0).astype(np.uint8)
        
        # Resize and normalize
        image_slice = cv2.resize(image_slice, (256, 256))
        mask_slice = cv2.resize(mask_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        images.append(image_slice)
        masks.append(mask_slice)
    
    return np.array(images), np.array(masks)

def main():
    # Download and preprocess data
    if not os.path.exists("BraTS2020_TrainingData"):
        download_dataset()
    
    print("Preprocessing data...")
    images, masks = preprocess_data()
    
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    
    # Create model
    model = unet_model(input_shape=(256, 256, 1))
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        "brain_tumor_segmentation.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    
    plt.savefig("training_history.png")
    print("Training complete. Model saved as brain_tumor_segmentation.h5")

if __name__ == "__main__":
    main()
