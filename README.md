# Image Recommendation System

A content-based image recommendation system that suggests similar images based on visual features using deep learning and computer vision techniques. This system can recommend artworks based on visual similarity to a query image.

## Features

- **Content-Based Filtering**: Recommends images based on visual similarity
- **Deep Feature Extraction**: Uses VGG16/ResNet50 for extracting high-level features
- **Color Analysis**: Incorporates color histograms for better recommendations
- **Memory Efficient**: Optimized for low-memory environments
- **Easy to Use**: Simple command-line interface

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- At least 4GB RAM (8GB recommended)
- 1GB free disk space

## Quick Start

1. **Clone and setup**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd image-recommendation-system
   
   # Create and activate virtual environment (Windows)
   python -m venv venv
   .\venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## Basic Usage

### 1. Run the Pipeline

Process images and get recommendations in one step:
```bash
# Process images and get recommendations
python run_pipeline.py --query "path/to/your/image.jpg" --batch-size 4
```

### 2. Get Recommendations for Your Image

To get recommendations for a specific image:
```bash
# Get top 5 similar images
python run_pipeline.py --query "4612745.jpg" --top-k 5

# Specify custom directories
python run_pipeline.py --query "images/my_artwork.jpg" --input-dir "data/raw" --output-dir "data/processed" --results-dir "my_results"
```

### 3. Process Images in Batches

For large datasets, process images in smaller batches:
```bash
# Process images with a smaller batch size (uses less memory)
python run_pipeline.py --batch-size 2
```

## Advanced Options

```
python run_pipeline.py --help

usage: run_pipeline.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                      [--results-dir RESULTS_DIR] [--query QUERY] [--top-k TOP_K]
                      [--batch-size BATCH_SIZE] [--skip-processing]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        Directory containing raw images (default: data/raw)
  --output-dir OUTPUT_DIR
                        Directory to save processed data (default: data/processed)
  --results-dir RESULTS_DIR
                        Directory to save results (default: results)
  --query QUERY         Path to query image for recommendations
  --top-k TOP_K         Number of recommendations to return (default: 5)
  --batch-size BATCH_SIZE
                        Batch size for processing images (default: 4)
  --skip-processing     Skip image processing if features already exist
```

## Example

1. **Process your image collection**:
   ```bash
   python run_pipeline.py --input-dir "my_images" --output-dir "processed_features"
   ```

2. **Get recommendations**:
   ```bash
   python run_pipeline.py --query "my_images/artwork.jpg" --top-k 3
   ```

## Results

Results are saved in the specified `--results-dir` (default: `results/`):
- `recommendations/`: Directory containing visual comparisons
- `features/`: Extracted features for faster subsequent runs
- `logs/`: Processing logs

## Troubleshooting

- **Memory Issues**: Try reducing the batch size (e.g., `--batch-size 2`)
- **CUDA Out of Memory**: Set environment variable: `set TF_ENABLE_ONEDNN_OPTS=0`
- **Slow Processing**: Ensure you're using a GPU if available

## Technical Details

- **Feature Extraction**: VGG16/ResNet50 CNN + Color Histograms
- **Similarity Metric**: Cosine Similarity
- **Memory Optimization**: Batch processing and memory cleanup
- **Supported Formats**: JPG, PNG, BMP


