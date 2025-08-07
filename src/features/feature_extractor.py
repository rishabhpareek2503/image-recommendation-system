"""
Feature extraction implementation for the Image Recommendation System.

This module provides functionality to extract features from images using
pre-trained CNN models and other computer vision techniques.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model

class FeatureExtractor:
    """A class to handle feature extraction from images using various methods."""
    
    def __init__(self, model_name: str = 'vgg16', input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """Initialize the feature extractor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use ('vgg16' or 'resnet50')
            input_shape: Expected input shape of the model
        """
        self.model_name = model_name.lower()
        self.input_shape = input_shape
        self.model = self._load_model()
        self.feature_extractor = self._create_feature_extractor()
    
    def _load_model(self) -> Model:
        """Load the pre-trained model.
        
        Returns:
            Loaded Keras model
        """
        if self.model_name == 'vgg16':
            # Load VGG16 without the top (fully connected) layers to save memory
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
            # Add global average pooling to reduce the number of parameters
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            return Model(inputs=base_model.input, outputs=x)
            
        elif self.model_name == 'resnet50':
            # Load ResNet50 without the top (fully connected) layers
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
            # Use the existing average pooling layer
            return Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Choose 'vgg16' or 'resnet50'.")
    
    def _create_feature_extractor(self) -> Model:
        """Create a feature extraction model from the base model.
        
        Returns:
            Model that outputs feature vectors
        """
        return Model(inputs=self.model.input, outputs=self.model.layers[-1].output)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for the model.
        
        Args:
            image: Input image as a numpy array (H, W, 3)
            
        Returns:
            Preprocessed image
        """
        # Resize if necessary
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Convert to float32 if needed
        if image.dtype != np.float32:
            image = image.astype('float32')
        
        # Apply model-specific preprocessing
        if self.model_name == 'vgg16':
            return vgg_preprocess(image)
        elif self.model_name == 'resnet50':
            return resnet_preprocess(image)
        else:
            return image / 255.0  # Default normalization
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image.
        
        Args:
            image: Input image as a numpy array (H, W, 3)
            
        Returns:
            Feature vector
        """
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Add batch dimension
        batch = np.expand_dims(processed_image, axis=0)
        
        # Extract features
        features = self.feature_extractor.predict(batch, verbose=0)
        
        # Flatten the features
        return features.flatten()
    
    def extract_batch_features(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extract features from a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            batch_size: Number of images to process in each batch
            
        Returns:
            Array of feature vectors
        """
        # Preprocess all images
        processed_images = np.array([self.preprocess_image(img) for img in images])
        
        # Extract features in batches
        features = []
        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i + batch_size]
            batch_features = self.feature_extractor.predict(batch, verbose=0)
            # Reshape to (batch_size, -1) to flatten the features
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
            features.append(batch_features)
        
        return np.vstack(features)
    
    def extract_color_histogram(self, image: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
        """Extract a color histogram from an image.
        
        Args:
            image: Input image as a numpy array (H, W, 3)
            bins: Number of bins for each color channel
            
        Returns:
            Flattened color histogram
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Compute the color histogram
        hist = cv2.calcHist(
            [hsv], 
            [0, 1, 2],  # Use all channels
            None,  # No mask
            bins,  # Number of bins per channel
            [0, 180, 0, 256, 0, 256]  # Range for each channel
        )
        
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist


def get_feature_extractor(model_name: str = 'vgg16', input_shape: Tuple[int, int, int] = (224, 224, 3)) -> FeatureExtractor:
    """Factory function to get a feature extractor instance.
    
    Args:
        model_name: Name of the pre-trained model to use
        input_shape: Expected input shape of the model
        
    Returns:
        FeatureExtractor instance
    """
    return FeatureExtractor(model_name=model_name, input_shape=input_shape)


def extract_features_batch(images: List[np.ndarray], model_name: str = 'vgg16', 
                         batch_size: int = 32) -> Dict[str, np.ndarray]:
    """Extract features from a batch of images.
    
    Args:
        images: List of input images as numpy arrays
        model_name: Name of the pre-trained model to use
        batch_size: Number of images to process in each batch
        
    Returns:
        Dictionary containing:
            - 'cnn_features': Array of CNN features
            - 'color_histograms': Array of color histograms
    """
    # Initialize feature extractor
    extractor = FeatureExtractor(model_name=model_name)
    
    # Extract CNN features
    cnn_features = extractor.extract_batch_features(images, batch_size=batch_size)
    
    # Extract color histograms
    color_hists = np.array([extractor.extract_color_histogram(img) for img in images])
    
    return {
        'cnn_features': cnn_features,
        'color_histograms': color_hists
    }
