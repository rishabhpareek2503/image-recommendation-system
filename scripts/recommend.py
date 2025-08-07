"""
Image Recommendation Script for the Image Recommendation System.

This script provides recommendations for similar images based on a query image.
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.recommender import get_recommender
from src.features.feature_extractor import FeatureExtractor
from src.utils.utils import load_image, display_images

def load_features(features_dir):
    """
    Load precomputed features and metadata.
    
    Args:
        features_dir: Directory containing the processed features
        
    Returns:
        Tuple of (cnn_features, color_histograms, metadata)
    """
    features_dir = Path(features_dir)
    
    # Load features
    cnn_features = np.load(features_dir / "cnn_features.npy")
    color_histograms = np.load(features_dir / "color_histograms.npy")
    
    # Load metadata
    with open(features_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return cnn_features, color_histograms, metadata

def get_recommendations(query_image_path, features_dir, top_k=5):
    """
    Get image recommendations based on a query image.
    
    Args:
        query_image_path: Path to the query image
        features_dir: Directory containing the processed features
        top_k: Number of recommendations to return
        
    Returns:
        Tuple of (recommended_images, recommended_paths, scores)
    """
    # Load features and metadata
    cnn_features, color_histograms, metadata = load_features(features_dir / "features")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(model_name='vgg16')
    
    # Load and process query image
    query_image = load_image(query_image_path, target_size=(224, 224))
    if query_image is None:
        print(f"Error: Could not load query image {query_image_path}")
        return [], [], []
    
    # Print debug info
    print(f"Query image shape: {query_image.shape}")
    print(f"Loaded {len(metadata)} images with features")
    
    # Extract features for query image
    try:
        print("Extracting features from query image...")
        query_cnn = extractor.extract_features(query_image).reshape(1, -1)
        query_color = extractor.extract_color_histogram(query_image).reshape(1, -1)
        
        print(f"Query CNN features shape: {query_cnn.shape}")
        print(f"Query color features shape: {query_color.shape}")
        print(f"Database CNN features shape: {cnn_features.shape}")
        print(f"Database color features shape: {color_histograms.shape}")
        
        # Get recommendations
        recommender = get_recommender(
            method='content_based',
            feature_weights={
                'cnn_features': 1.0,  # Let's focus on CNN features first
                'color_histograms': 0.0
            }
        )
        
        # Print feature statistics
        print("\nFeature statistics before normalization:")
        print(f"Query CNN - Shape: {query_cnn.shape}, Min: {np.min(query_cnn):.4f}, Max: {np.max(query_cnn):.4f}, Mean: {np.mean(query_cnn):.4f}")
        print(f"Query Color - Shape: {query_color.shape}, Min: {np.min(query_color):.4f}, Max: {np.max(query_color):.4f}, Mean: {np.mean(query_color):.4f}")
        print(f"DB CNN - Shape: {cnn_features.shape}, Min: {np.min(cnn_features):.4f}, Max: {np.max(cnn_features):.4f}, Mean: {np.mean(cnn_features):.4f}")
        print(f"DB Color - Shape: {color_histograms.shape}, Min: {np.min(color_histograms):.4f}, Max: {np.max(color_histograms):.4f}, Mean: {np.mean(color_histograms):.4f}")
        
        # Create feature sets for similarity calculation
        query_features = {
            'cnn_features': query_cnn.astype(np.float32),
            'color_histograms': query_color.astype(np.float32)
        }
        
        target_features = {
            'cnn_features': cnn_features.astype(np.float32),
            'color_histograms': color_histograms.astype(np.float32)
        }
        
        # Initialize recommender with weights
        recommender = get_recommender(
            method='content_based',
            feature_weights={
                'cnn_features': 0.8,  # Higher weight for CNN features
                'color_histograms': 0.2  # Lower weight for color histograms
            }
        )
        
        # Calculate similarity scores
        print("\nCalculating similarities...")
        similarity_scores = recommender.calculate_similarity(
            query_features=query_features,
            target_features=target_features
        )
        
        # Debug: Print score statistics
        print(f"\nScore statistics:")
        print(f"Min: {np.min(similarity_scores):.6f}")
        print(f"Max: {np.max(similarity_scores):.6f}")
        print(f"Mean: {np.mean(similarity_scores):.6f}")
        print(f"Median: {np.median(similarity_scores):.6f}")
        
        # Print top 10 scores and indices for debugging
        top10_idx = np.argsort(similarity_scores)[::-1][:10]
        print("\nTop 10 matches:")
        for i, idx in enumerate(top10_idx, 1):
            print(f"  {i}. Index: {idx}, Score: {similarity_scores[idx]:.6f}")
        
        # Get top-k recommendations (excluding the query image if it's in the dataset)
        top_k_indices = np.argsort(similarity_scores)[::-1][:top_k + 1]
        
    except Exception as e:
        print(f"Error during feature extraction or similarity calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], [], []
    
    # Get recommended images and scores
    recommended_images = []
    recommended_paths = []
    scores = []
    
    for idx in top_k_indices:
        img_path = Path(features_dir).parent / 'raw' / metadata[idx]['path']
        img = load_image(str(img_path))
        if img is not None:
            recommended_images.append(img)
            recommended_paths.append(str(img_path))
            scores.append(similarity_scores[idx])
    
    return recommended_images, recommended_paths, scores

def display_results(query_image_path, recommended_images, scores, save_path=None):
    """
    Display query image and recommendations.
    
    Args:
        query_image_path: Path to the query image
        recommended_images: List of recommended images
        scores: List of similarity scores
        save_path: Path to save the results (optional)
    """
    # Load query image
    query_img = load_image(query_image_path)
    if query_img is None:
        print(f"Error: Could not load query image {query_image_path}")
        return
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Display query image
    plt.subplot(1, len(recommended_images) + 1, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    
    # Display recommendations
    for i, (img, score) in enumerate(zip(recommended_images, scores), 2):
        plt.subplot(1, len(recommended_images) + 1, i)
        plt.imshow(img)
        plt.title(f"Score: {score:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Results saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    import random
    from pathlib import Path
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Get image recommendations')
    parser.add_argument('--query', type=str, default=None,
                       help='Path to query image (random if not specified)')
    parser.add_argument('--features-dir', type=str, default='data/processed',
                       help='Directory containing processed features')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of recommendations to return')
    parser.add_argument('--show-results', action='store_true',
                       help='Show results in a window')
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Get query image
    if args.query:
        query_image_path = args.query
    else:
        # Get a random image from the dataset
        raw_images_dir = Path(args.features_dir).parent / 'raw'
        image_paths = list(raw_images_dir.glob('*.*'))
        if not image_paths:
            print(f"No images found in {raw_images_dir}")
            sys.exit(1)
        query_image_path = str(random.choice(image_paths))
    
    print(f"Using query image: {query_image_path}")
    
    # Get recommendations
    recommended_images, recommended_paths, scores = get_recommendations(
        query_image_path=query_image_path,
        features_dir=Path(args.features_dir),
        top_k=args.top_k
    )
    
    if not recommended_images:
        print("No recommendations found.")
        sys.exit(1)
    
    # Display and save results
    result_path = results_dir / "recommendations.png"
    display_results(
        query_image_path=query_image_path,
        recommended_images=recommended_images,
        scores=scores,
        save_path=str(result_path)
    )
    
    # Print recommended image paths
    print("\nRecommended images:")
    for path, score in zip(recommended_paths, scores):
        print(f"- {path} (Score: {score:.3f})")
