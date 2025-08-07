"""
Image Recommendation System Pipeline

This script runs the complete image recommendation pipeline:
1. Processes images and extracts features
2. Generates recommendations based on a query image
3. Displays and saves the results
"""

import os
import sys
import argparse
from pathlib import Path
import random

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from scripts.process_images import process_images
from scripts.recommend import get_recommendations, display_results
from utils.utils import load_image, get_image_files

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Image Recommendation System Pipeline')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Directory containing raw images')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--query', type=str, default=None,
                       help='Path to query image (random if not specified)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of recommendations to return')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Number of images to process in each batch')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip the image processing step')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process images and extract features
    if not args.skip_processing:
        print("\n" + "="*50)
        print("Step 1: Processing images and extracting features...")
        print("="*50)
        process_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    else:
        print("Skipping image processing step...")
    
    # Step 2: Get a query image
    print("\n" + "="*50)
    print("Step 2: Preparing query image...")
    print("="*50)
    
    if args.query:
        query_image_path = args.query
        print(f"Using specified query image: {query_image_path}")
    else:
        # Get a random image from the input directory
        image_paths = get_image_files(args.input_dir)
        if not image_paths:
            print(f"No images found in {args.input_dir}")
            return
        
        query_image_path = random.choice(image_paths)
        print(f"Selected random query image: {query_image_path}")
    
    # Verify query image exists
    if not os.path.exists(query_image_path):
        print(f"Error: Query image not found: {query_image_path}")
        return
    
    # Step 3: Get recommendations
    print("\n" + "="*50)
    print("Step 3: Generating recommendations...")
    print("="*50)
    
    recommended_images, recommended_paths, scores = get_recommendations(
        query_image_path=query_image_path,
        features_dir=Path(args.output_dir),
        top_k=args.top_k
    )
    
    if not recommended_images:
        print("No recommendations found.")
        return
    
    # Step 4: Display and save results
    print("\n" + "="*50)
    print("Step 4: Saving results...")
    print("="*50)
    
    # Save results
    result_path = Path(args.results_dir) / "recommendations.png"
    display_results(
        query_image_path=query_image_path,
        recommended_images=recommended_images,
        scores=scores,
        save_path=str(result_path)
    )
    
    # Print recommended image paths
    print("\nRecommended images:")
    for i, (path, score) in enumerate(zip(recommended_paths, scores), 1):
        print(f"{i}. {path} (Score: {score:.3f})")
    
    print(f"\nResults saved to: {result_path}")

if __name__ == "__main__":
    main()
