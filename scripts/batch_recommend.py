"""
Batch Image Recommendation System

This script provides batch recommendations based on multiple liked images.
"""

import os
import json
import numpy as np
from pathlib import Path
import random
from typing import List, Dict, Tuple

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.recommender import get_recommender
from src.features.feature_extractor import FeatureExtractor
from src.utils.utils import load_image, display_images

def load_features(features_dir: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load precomputed features and metadata."""
    features_dir = Path(features_dir)
    
    # Load features
    cnn_features = np.load(features_dir / "cnn_features.npy")
    color_histograms = np.load(features_dir / "color_histograms.npy")
    
    # Load metadata
    with open(features_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return cnn_features, color_histograms, metadata

def get_batch_recommendations(liked_images: List[str], 
                           features_dir: str, 
                           batch_size: int = 5) -> List[Dict]:
    """
    Get batch recommendations based on liked images.
    
    Args:
        liked_images: List of paths to liked images
        features_dir: Directory containing the processed features
        batch_size: Number of recommendations to return
        
    Returns:
        List of recommended images with their scores and paths
    """
    # Load features and metadata
    cnn_features, color_histograms, metadata = load_features(features_dir / "features")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(model_name='vgg16')
    
    # Process liked images
    liked_features = []
    for img_path in liked_images:
        # Load and process image
        img = load_image(img_path, target_size=(224, 224))
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
            
        # Extract features
        cnn_feat = extractor.extract_features(img).reshape(1, -1)
        color_feat = extractor.extract_color_histogram(img).reshape(1, -1)
        liked_features.append((cnn_feat, color_feat))
    
    if not liked_features:
        print("Error: No valid liked images provided")
        return []
    
    # Average features of liked images
    avg_cnn = np.mean(np.vstack([f[0] for f in liked_features]), axis=0, keepdims=True)
    avg_color = np.mean(np.vstack([f[1] for f in liked_features]), axis=0, keepdims=True)
    
    # Get recommendations
    recommender = get_recommender(
        method='content_based',
        feature_weights={
            'cnn_features': 0.8,  # Higher weight for visual features
            'color_histograms': 0.2
        }
    )
    
    # Calculate similarity scores
    similarity_scores = recommender.calculate_similarity(
        query_features={
            'cnn_features': avg_cnn,
            'color_histograms': avg_color
        },
        target_features={
            'cnn_features': cnn_features,
            'color_histograms': color_histograms
        }
    )
    
    # Get top recommendations (excluding already liked images)
    all_paths = [str(Path(features_dir).parent / 'raw' / m['path']) for m in metadata]
    liked_paths = set([str(Path(p).resolve()) for p in liked_images])
    
    # Sort by score, filter out already liked, and get top N
    sorted_indices = np.argsort(similarity_scores)[::-1]
    recommendations = []
    
    for idx in sorted_indices:
        if len(recommendations) >= batch_size + len(liked_images):
            break
            
        img_path = all_paths[idx]
        if img_path not in liked_paths:
            recommendations.append({
                'path': img_path,
                'score': float(similarity_scores[idx])
            })
    
    return recommendations[:batch_size]

def get_recommendation_batches(liked_images, features_dir, batch_size=5):
    """
    Get recommendations in batches based on liked images.
    
    Args:
        liked_images: List of exactly 10 liked image paths
        features_dir: Directory containing processed features
        batch_size: Number of recommendations per batch (default: 5)
        
    Returns:
        List of recommendation batches, each containing batch_size recommendations
    """
    if len(liked_images) != 10:
        raise ValueError("Exactly 10 liked images are required")
    
    # Get initial recommendations based on all 10 liked images
    all_recommendations = get_batch_recommendations(
        liked_images=liked_images,
        features_dir=Path(features_dir),
        batch_size=20  # Get more recommendations to create multiple batches
    )
    
    # Split into batches
    batches = [all_recommendations[i:i + batch_size] 
               for i in range(0, len(all_recommendations), batch_size)]
    
    return batches

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Get batch image recommendations')
    parser.add_argument('--liked', nargs=10, required=True,
                       help='Exactly 10 liked image paths')
    parser.add_argument('--features-dir', type=str, default='data/processed',
                       help='Directory containing processed features')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    try:
        # Get recommendation batches
        batches = get_recommendation_batches(
            liked_images=args.liked,
            features_dir=args.features_dir
        )
        
        # Display and save results
        with open(results_dir / 'batch_recommendations.txt', 'w') as f:
            for batch_num, batch in enumerate(batches, 1):
                print(f"\n=== Batch {batch_num} ===")
                f.write(f"\n=== Batch {batch_num} ===\n")
                
                for i, rec in enumerate(batch, 1):
                    rec_line = f"{i}. {rec['path']} (Score: {rec['score']:.4f})"
                    print(rec_line)
                    f.write(rec_line + '\n')
        
        print(f"\nResults saved to {results_dir / 'batch_recommendations.txt'}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
