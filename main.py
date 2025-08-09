import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import os

from util import set_background
from model import unet_model, postprocess_mask


set_background('./bg.jpeg')

# Load model
model = unet_model()
weights_path = 'brain_tumor_segmentation.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    st.success("Tumor detection model loaded successfully")
else:
    st.warning("Trained weights not found. Using untrained model. Please train the model first for accurate results.")

# set title
st.title('Brain MRI tumor detection - CNN Model')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('Upload MRI scan image', type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

# process image
if file:
    # Load image
    original_img = Image.open(file).convert('RGB')
    
    # Preprocess for model
    input_img = original_img.resize((256, 256))
    image_array = np.array(input_img) / 255.0
    input_tensor = tf.expand_dims(image_array, axis=0)
    
    # Predict tumor mask
    pred_mask = model.predict(input_tensor)[0]
    
    # Postprocess mask
    refined_mask = postprocess_mask(pred_mask)
    refined_mask = cv2.resize(refined_mask, original_img.size)
    
    # Convert mask to RGBA for visualization
    mask_rgba = np.zeros((refined_mask.shape[0], refined_mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[refined_mask == 1] = [255, 0, 0, 150]  # Red with transparency
    
    # Create overlay - ensure mask matches original image dimensions
    overlay = Image.fromarray(mask_rgba)
    if overlay.size != original_img.size:
        overlay = overlay.resize(original_img.size)
    
    result_img = Image.alpha_composite(
        original_img.convert("RGBA"), 
        overlay
    ).convert("RGB")
    
    # Resize input image for display
    display_img = original_img.resize((256, 256))
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(display_img, use_column_width=True)
    with col2:
        st.subheader("Tumor Detection")
        st.image(result_img, use_column_width=True)
        
    # Show tumor area percentage
    tumor_area = np.sum(refined_mask) / (refined_mask.shape[0] * refined_mask.shape[1]) * 100
    st.info(f"Detected tumor area: {tumor_area:.2f}% of image")
