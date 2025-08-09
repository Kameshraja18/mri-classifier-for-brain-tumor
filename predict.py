import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_image(img_path, img_size=(224, 224)):
    """Load and preprocess a single image"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

def predict_tumor(model, img_path):
    """Predict whether an image contains a brain tumor"""
    img = load_image(img_path)
    prediction = model.predict(img)[0][0]
    
    # Convert probability to class
    if prediction >= 0.5:
        class_label = "Tumor Detected"
        confidence = prediction
    else:
        class_label = "No Tumor"
        confidence = 1 - prediction
    
    return class_label, confidence

def main():
    # Check if model exists
    model_path = "brain_tumor_classifier.h5"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using: python train_classifier.py")
        return
    
    # Load trained model
    print("Loading trained model...")
    model = load_model(model_path)
    
    # Test on sample images
    test_dirs = [
        "BraTS2020_TrainingData/no",
        "BraTS2020_TrainingData/yes"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\nTesting images from: {test_dir}")
            image_files = [f for f in os.listdir(test_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
            
            for img_file in image_files:
                img_path = os.path.join(test_dir, img_file)
                try:
                    label, confidence = predict_tumor(model, img_path)
                    print(f"{img_file}: {label} (confidence: {confidence:.4f})")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")

if __name__ == "__main__":
    main()
