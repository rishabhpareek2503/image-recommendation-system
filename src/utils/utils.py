"""
Utility functions for the Image Recommendation System.

This module provides various utility functions for the Image Recommendation System.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image, ExifTags
from pathlib import Path

def create_directory(directory: str) -> None:
    """Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory to create
    """
    os.makedirs(directory, exist_ok=True)

def get_file_extension(filename: str) -> str:
    """Get the lowercase file extension without the dot.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension in lowercase
    """
    return Path(filename).suffix.lower().lstrip('.')

def is_image_file(filename: str, extensions: set = None) -> bool:
    """Check if a file is an image based on its extension.
    
    Args:
        filename: Name of the file to check
        extensions: Set of valid image extensions (default: {'jpg', 'jpeg', 'png', 'bmp'})
        
    Returns:
        True if the file is an image, False otherwise
    """
    if extensions is None:
        extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp'}
    return get_file_extension(filename) in extensions

def get_image_files(directory: str, recursive: bool = True) -> List[str]:
    """Get a list of image files in a directory.
    
    Args:
        directory: Directory to search for images
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of paths to image files
    """
    directory = Path(directory)
    if recursive:
        return [str(p) for p in directory.rglob('*') if p.is_file() and is_image_file(p.name)]
    return [str(p) for p in directory.glob('*') if p.is_file() and is_image_file(p.name)]

def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Load an image from disk.
    
    Args:
        image_path: Path to the image file
        target_size: Optional target size as (height, width)
        
    Returns:
        Image as a numpy array in RGB format, or None if loading fails
    """
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if target size is provided
        if target_size is not None:
            img = cv2.resize(img, (target_size[1], target_size[0]))
            
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """Save an image to disk.
    
    Args:
        image: Image as a numpy array
        output_path: Path to save the image
        quality: JPEG quality (1-100)
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert from RGB to BGR if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Save the image
        return cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract metadata from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing image metadata
    """
    metadata = {
        'path': image_path,
        'filename': os.path.basename(image_path),
        'size_bytes': os.path.getsize(image_path),
        'created': os.path.getctime(image_path),
        'modified': os.path.getmtime(image_path),
        'dimensions': None,
        'format': None,
        'exif': {}
    }
    
    try:
        with Image.open(image_path) as img:
            # Get basic image information
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            metadata['dimensions'] = img.size  # (width, height)
            
            # Extract EXIF data if available
            if hasattr(img, '_getexif') and img._getexif() is not None:
                for tag, value in img._getexif().items():
                    if tag in ExifTags.TAGS:
                        metadata['exif'][ExifTags.TAGS[tag]] = value
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
    
    return metadata

def display_images(images: List[np.ndarray], titles: List[str] = None, figsize: Tuple[int, int] = (15, 5),
                 save_path: Optional[str] = None) -> None:
    """Display multiple images in a row with optional titles.
    
    Args:
        images: List of images to display (numpy arrays)
        titles: Optional list of titles for each image
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    n = len(images)
    if titles is None:
        titles = [''] * n
    
    plt.figure(figsize=figsize)
    
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, n, i)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
