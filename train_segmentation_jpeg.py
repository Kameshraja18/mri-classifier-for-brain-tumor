import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from model import unet_model
import matplotlib.pyplot as plt

def create_synthetic_masks(images):
    """Create synthetic circular masks for JPEG brain tumor images"""
    masks = []
    for img in images:
        # Create a simple circular mask as synthetic ground truth
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create a circular region in the center as tumor
        center_x, center_y = w // 2, h // 2
        radius = min(h, w) // 6
        
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)
        
        # Add some noise to make it more realistic
        noise = np.random.random((h, w)) < 0.1
        mask = np.clip(mask + noise, 0, 1)
        
        masks.append(mask)
    
    return np.array(masks)

def load_jpeg_data():
    """Load JPEG images for segmentation training"""
    images = []
    masks = []
    
    # Load tumor images
    tumor_dir = "BraTS2020_TrainingData/yes"
    if os.path.exists(tumor_dir):
        tumor_files = [f for f in os.listdir(tumor_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]  # Limit for demo
        
        for file in tumor_files:
            img_path = os.path.join(tumor_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img = img / 255.0
                images.append(img)
    
    # Load non-tumor images
    no_tumor_dir = "BraTS2020_TrainingData/no"
    if os.path.exists(no_tumor_dir):
        no_tumor_files = [f for f in os.listdir(no_tumor_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]  # Limit for demo
        
        for file in no_tumor_files:
            img_path = os.path.join(no_tumor_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img = img / 255.0
                images.append(img)
    
    images = np.array(images)
    masks = create_synthetic_masks(images)
    
    # Add channel dimensions
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)
    
    return images, masks

def main():
    print("Loading JPEG data for segmentation...")
    
    # Load data
    images, masks = load_jpeg_data()
    
    if len(images) == 0:
        print("No images found. Please check the dataset path.")
        return
    
    print(f"Loaded {len(images)} images for segmentation training")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    
    
    # In train_segmentation_jpeg.py, line 92
    model = unet_model(input_shape=(256, 256, 3))
    # Callbacks
    checkpoint = ModelCheckpoint(
        "brain_tumor_segmentation_jpeg.h5",
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
    print("Training segmentation model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
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
    
    plt.savefig("segmentation_training_history.png")
    print("Training complete. Model saved as brain_tumor_segmentation_jpeg.h5")

if __name__ == "__main__":
    main()
