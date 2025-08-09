import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os

def unet_model(input_shape=(256, 256, 3)):
    """U-Net model with EfficientNetB3 backbone for brain tumor segmentation"""
    # Encoder (EfficientNetB3 backbone) - use local weights
    weights_path = "efficientnetb3_notop.h5"
    if not os.path.exists(weights_path):
        # Download weights if not present
        url = "https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5"
        import urllib.request
        urllib.request.urlretrieve(url, weights_path)
    
    base_model = EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights=None  # We'll load manually
    )
    base_model.load_weights(weights_path)
    
    # Freeze encoder layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Bridge
    bridge = base_model.get_layer('top_activation').output
    
    # Decoder
    u = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bridge)
    u = layers.concatenate([u, base_model.get_layer('block6a_expand_activation').output])
    u = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u)
    u = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u)
    
    u = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u)
    u = layers.concatenate([u, base_model.get_layer('block4a_expand_activation').output])
    u = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u)
    u = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u)
    
    u = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u)
    u = layers.concatenate([u, base_model.get_layer('block3a_expand_activation').output])
    u = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u)
    u = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u)
    
    u = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u)
    u = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u)
    u = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u)
    
    # Output layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(u)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    return model

def postprocess_mask(mask):
    """Post-process segmentation mask using connected components"""
    mask = mask.squeeze()
    mask = (mask > 0.5).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    
    # Keep only the largest component
    if num_labels > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        max_label = np.argmax(sizes) + 1
        mask = (labels == max_label).astype(np.uint8)
    else:
        mask = np.zeros_like(mask, dtype=np.uint8)
    
    return mask
