"""
Main module for the Image Recommendation System.

This module demonstrates how to use the recommendation system to find similar images
based on a set of liked images.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import our modules
from data.data_loader import ImageDataLoader, create_mock_dataset, load_image_data
from features.feature_extractor import FeatureExtractor, extract_features_batch
from models.recommender import get_recommender
from utils.utils import get_image_files, load_image, save_image
from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    FEATURE_EXTRACTION_SETTINGS, RECOMMENDATION_SETTINGS
)

class ImageRecommender:
    """A class to handle the end-to-end image recommendation process."""
    
    def __init__(self, data_dir: str = None):
        """Initialize the image recommender.
        
        Args:
            data_dir: Directory containing the image dataset
        """
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.feature_extractor = None
        self.recommender = None
        self.data_loader = ImageDataLoader(str(self.data_dir))
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_data(self, use_mock: bool = False) -> Dict[str, Any]:
        """Load the image data.
        
        Args:
            use_mock: Whether to use mock data for testing
            
        Returns:
            Dictionary containing the loaded data
        """
        if use_mock:
            print("Using mock dataset...")
            return create_mock_dataset()
        
        print(f"Loading images from {self.data_dir}...")
        return load_image_data(str(self.data_dir))
    
    def initialize_models(self, model_name: str = None):
        """Initialize the feature extractor and recommender models.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        model_name = model_name or FEATURE_EXTRACTION_SETTINGS['cnn']['model_name']
        print(f"Initializing {model_name} feature extractor...")
        self.feature_extractor = FeatureExtractor(model_name=model_name)
        
        print("Initializing recommender...")
        self.recommender = get_recommender(
            method='content_based',
            feature_weights=RECOMMENDATION_SETTINGS['content_based']['feature_weights']
        )
    
    def extract_features(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract features from a list of images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            Dictionary of extracted features
        """
        if not self.feature_extractor:
            self.initialize_models()
        
        print("Extracting features...")
        return extract_features_batch(
            images, 
            model_name=self.feature_extractor.model_name
        )
    
    def get_recommendations(self, 
                          query_features: Dict[str, np.ndarray],
                          target_features: Dict[str, np.ndarray],
                          k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Get recommendations based on query features.
        
        Args:
            query_features: Features of the query images
            target_features: Features of the target images to recommend from
            k: Number of recommendations to return
            
        Returns:
            Tuple of (indices, scores) for the top-k recommendations
        """
        if not self.recommender:
            self.initialize_models()
        
        print("Generating recommendations...")
        return self.recommender.recommend(
            query_features, 
            target_features, 
            k=k
        )
    
    def run(self, query_image_paths: List[str] = None, k: int = 5, use_mock: bool = False):
        """Run the recommendation pipeline.
        
        Args:
            query_image_paths: List of paths to query images
            k: Number of recommendations to return
            use_mock: Whether to use mock data for testing
        """
        # Load data
        data = self.load_data(use_mock=use_mock)
        
        # If no query images provided, use the liked images
        if not query_image_paths and 'liked_indices' in data:
            query_image_paths = [data['image_paths'][i] for i in data['liked_indices']]
        
        if not query_image_paths:
            raise ValueError("No query images provided and no liked images found in the dataset.")
        
        # Load query images
        print(f"Loading {len(query_image_paths)} query images...")
        query_images = []
        valid_query_paths = []
        
        for path in query_image_paths:
            img = load_image(path)
            if img is not None:
                query_images.append(img)
                valid_query_paths.append(path)
        
        if not query_images:
            raise ValueError("No valid query images could be loaded.")
        
        # Extract features for query images
        print("Extracting features from query images...")
        query_features = self.extract_features(query_images)
        
        # If using mock data, we already have the target features
        if use_mock:
            print("Using pre-computed target features from mock data...")
            target_features = {
                'cnn_features': np.random.rand(len(data['images']), 512),  # Mock CNN features
                'color_histograms': np.random.rand(len(data['images']), 512)  # Mock color features
            }
        else:
            # Extract features for all target images
            print(f"Extracting features from {len(data['images'])} target images...")
            target_features = self.extract_features(data['images'])
        
        # Get recommendations
        indices, scores = self.get_recommendations(
            query_features, 
            target_features, 
            k=k
        )
        
        # Display results
        print("\n=== Recommendations ===")
        for i, (idx, score) in enumerate(zip(indices, scores)):
            print(f"{i+1}. Image {data['image_paths'][idx]} (Score: {score:.4f})")
        
        return {
            'indices': indices,
            'scores': scores,
            'image_paths': [data['image_paths'][i] for i in indices]
        }

def main():
    """Main function to run the image recommender from the command line."""
    parser = argparse.ArgumentParser(description='Image Recommendation System')
    parser.add_argument('--data-dir', type=str, default=str(RAW_DATA_DIR),
                        help='Directory containing the image dataset')
    parser.add_argument('--query', type=str, nargs='+',
                        help='Paths to query images')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of recommendations to return')
    parser.add_argument('--use-mock', action='store_true',
                        help='Use mock data for testing')
    
    args = parser.parse_args()
    
    # Initialize the recommender
    recommender = ImageRecommender(data_dir=args.data_dir)
    
    # Run the recommendation
    recommender.run(
        query_image_paths=args.query,
        k=args.top_k,
        use_mock=args.use_mock
    )

if __name__ == '__main__':
    main()
