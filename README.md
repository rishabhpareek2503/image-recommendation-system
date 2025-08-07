# Image Recommendation System

A content-based image recommendation system that suggests similar images based on visual features using deep learning and computer vision techniques. This system is designed to recommend artworks from a collection based on user preferences.

## Features

- **Content-Based Filtering**: Recommends images based on visual similarity
- **Deep Feature Extraction**: Uses VGG16 for extracting high-level features
- **Color Analysis**: Incorporates color histograms for better recommendations
- **Batch Processing**: Can process multiple liked images to generate recommendations
- **Scalable**: Designed to work with large image collections

## Project Structure

```
image-recommendation-system/
├── config/               # Configuration files
├── data/                 # Data storage
│   ├── raw/              # Raw image data (259 Alfred Sisley paintings)
│   └── processed/        # Processed features and metadata
├── results/              # Output results and visualizations
├── src/                  # Source code
│   ├── __init__.py
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature extraction modules
│   ├── models/           # Recommendation models
│   └── utils/            # Utility functions
├── scripts/              # Executable scripts
│   ├── process_images.py # Process images and extract features
│   ├── recommend.py      # Single image recommendation
│   └── batch_recommend.py# Batch recommendations
├── tests/                # Unit tests
├── .gitignore
├── requirements.txt      # Python dependencies
├── run_pipeline.py       # Full pipeline execution
└── test_batch_recommendations.py  # Test script for batch recommendations
```

## Setup Instructions

1. **Prerequisites**
   - Python 3.7+
   - pip (Python package manager)

2. **Clone and setup**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd image-recommendation-system
   
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # OR
   source venv/bin/activate  # macOS/Linux
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## Usage

### 1. Process Images
Extract features from the image dataset:
```bash
python scripts/process_images.py --input-dir data/raw --output-dir data/processed
```

### 2. Single Image Recommendation
Get recommendations based on a single query image:
```bash
python scripts/recommend.py --query data/raw/Alfred_Sisley_1.jpg --features-dir data/processed --top-k 5
```

### 3. Batch Recommendations
Get recommendations based on multiple liked images:
```bash
python scripts/batch_recommend.py --liked "data/raw/Alfred_Sisley_1.jpg" "data/raw/Alfred_Sisley_2.jpg" --batch-size 5
```


## Results

Results are saved in the `results/` directory, including:
- `recommendations.png`: Visual comparison of query and recommended images
- `batch_recommendations.txt`: Text file with recommended image paths and scores

## Technical Details

- **Feature Extraction**: VGG16 CNN + Color Histograms
- **Similarity Metric**: Cosine Similarity
- **Feature Weights**: 80% CNN features, 20% color features
- **Batch Processing**: Supports processing multiple images for better recommendations


