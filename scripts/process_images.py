"""
Image Processing Script for the Image Recommendation System.

This script processes images from the raw directory, extracts features,
and saves them for later use in recommendations.
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.feature_extractor import FeatureExtractor, extract_features_batch
from src.data.data_loader import ImageDataLoader
from src.utils.utils import get_image_files, load_image

def process_images(input_dir, output_dir, batch_size=8):
    """
    Process images and extract features.
    
    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save processed data
        batch_size: Number of images to process in each batch
    """
    # Create output directories
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    
    os.makedirs(features_dir, exist_ok=True)
    
    # Initialize data loader and feature extractor
    data_loader = ImageDataLoader(str(input_dir))
    extractor = FeatureExtractor(model_name='vgg16')
    
    # Get all image paths
    image_paths = get_image_files(str(input_dir))
    print(f"Found {len(image_paths)} images to process")
    
    # Process images in batches
    all_cnn_features = []
    all_color_histograms = []
    metadata = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        # Load images
        for path in batch_paths:
            img = load_image(path, target_size=(224, 224))
            if img is not None:
                batch_images.append(img)
                rel_path = str(Path(path).relative_to(input_dir))
                metadata.append({
                    'path': rel_path,
                    'original_size': img.shape[:2],
                    'filename': os.path.basename(path)
                })
            else:
                print(f"Warning: Could not load {path}")
        
        if not batch_images:
            continue
            
        # Extract features
        features = extract_features_batch(batch_images)
        all_cnn_features.append(features['cnn_features'])
        all_color_histograms.append(features['color_histograms'])
    
    # Combine features if any were extracted
    if all_cnn_features:
        # Convert lists to numpy arrays
        all_cnn_features = np.vstack(all_cnn_features)
        all_color_histograms = np.vstack(all_color_histograms)
        
        # Save features
        np.save(features_dir / "cnn_features.npy", all_cnn_features)
        np.save(features_dir / "color_histograms.npy", all_color_histograms)
        
        # Save metadata
        with open(features_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSuccessfully processed {len(metadata)} images")
        print(f"Features saved to {features_dir}")
    else:
        print("No features were extracted. Check if the images were loaded correctly.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images and extract features')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Directory containing raw images')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Number of images to process in each batch')
    
    args = parser.parse_args()
    
    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
