"""
Data loading and preprocessing utilities for the Image Recommendation System.
"""
import os
import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
from PIL import Image

class ImageDataLoader:
    """A class to handle loading and preprocessing of image data."""
    
    def __init__(self, data_dir: str = None):
        """Initialize the data loader with the directory containing image data.
        
        Args:
            data_dir: Path to the directory containing image data
        """
        self.data_dir = data_dir
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing the image (height, width)
            
        Returns:
            Preprocessed image as a numpy array
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image at {image_path}")
                
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (target_size[1], target_size[0]))
            
            # Normalize pixel values to [0, 1]
            image = image.astype('float32') / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def load_image_batch(self, image_paths: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Load a batch of images.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to load in each batch
            
        Returns:
            List of preprocessed images
        """
        images = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [self.load_image(path) for path in batch_paths]
            # Filter out None values (failed loads)
            batch_images = [img for img in batch_images if img is not None]
            images.extend(batch_images)
        return images
    
    def get_image_paths(self, directory: str = None) -> List[str]:
        """Get paths to all image files in the specified directory.
        
        Args:
            directory: Directory to search for images. Uses self.data_dir if None.
            
        Returns:
            List of paths to image files
        """
        if directory is None:
            if self.data_dir is None:
                raise ValueError("No directory specified and no default data_dir set")
            directory = self.data_dir
            
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

def create_mock_dataset(num_images: int = 200, image_size: Tuple[int, int] = (224, 224, 3)) -> Dict[str, Any]:
    """Create a mock dataset for testing and development.
    
    Args:
        num_images: Number of images to generate
        image_size: Size of each image (height, width, channels)
        
    Returns:
        Dictionary containing:
            - 'images': List of random image arrays
            - 'image_paths': List of mock image paths
            - 'liked_indices': Indices of liked images (10% of total)
            - 'metadata': Dictionary of mock metadata for each image
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate random images
    images = [np.random.randint(0, 256, size=image_size, dtype=np.uint8) 
              for _ in range(num_images)]
    
    # Generate mock image paths
    image_paths = [f"mock_image_{i:03d}.jpg" for i in range(num_images)]
    
    # Randomly select 10% of images as 'liked'
    num_liked = max(10, num_images // 10)
    liked_indices = np.random.choice(num_images, size=num_liked, replace=False).tolist()
    
    # Generate mock metadata
    categories = ["landscape", "portrait", "abstract", "wildlife", "urban", "food", "sports", "fashion"]
    metadata = {
        path: {
            "category": np.random.choice(categories),
            "brightness": np.random.uniform(0.1, 0.9),
            "contrast": np.random.uniform(0.8, 1.2),
            "is_liked": i in liked_indices
        }
        for i, path in enumerate(image_paths)
    }
    
    return {
        "images": images,
        "image_paths": image_paths,
        "liked_indices": liked_indices,
        "metadata": metadata
    }

def load_image_data(data_dir: str, batch_size: int = 32) -> Dict[str, Any]:
    """Load image data from a directory.
    
    Args:
        data_dir: Directory containing image files
        batch_size: Number of images to load in each batch
        
    Returns:
        Dictionary containing loaded images and metadata
    """
    loader = ImageDataLoader(data_dir)
    image_paths = loader.get_image_paths()
    images = loader.load_image_batch(image_paths, batch_size)
    
    # For now, create mock metadata
    metadata = {
        path: {
            "size": os.path.getsize(path),
            "modified": os.path.getmtime(path),
            "is_liked": False  # Will be updated based on user interaction
        }
        for path in image_paths
    }
    
    return {
        "images": images,
        "image_paths": image_paths,
        "liked_indices": [],  # Will be populated based on user interaction
        "metadata": metadata
    }
