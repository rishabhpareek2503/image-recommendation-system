""
Configuration settings for the Image Recommendation System.
"""
from pathlib import Path
import os

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model settings
DEFAULT_MODEL = 'vgg16'
INPUT_SHAPE = (224, 224, 3)  # Default input shape for the model
BATCH_SIZE = 32  # Batch size for processing images

# Feature extraction settings
FEATURE_EXTRACTION_SETTINGS = {
    'cnn': {
        'model_name': 'vgg16',  # Options: 'vgg16', 'resnet50'
        'layer_name': 'fc2',  # Layer to extract features from
        'pooling': 'avg',  # Pooling method: 'avg' or 'max'
    },
    'color': {
        'bins': (8, 8, 8),  # Number of bins for color histograms (H, S, V)
    },
    'texture': {
        'distance': 1,  # Distance for GLCM
        'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],  # Angles for GLCM
        'levels': 8,  # Number of gray levels for GLCM
    }
}

# Recommendation settings
RECOMMENDATION_SETTINGS = {
    'content_based': {
        'feature_weights': {
            'cnn_features': 0.7,
            'color_histograms': 0.3
        },
        'top_k': 10,  # Number of recommendations to return
    },
    'hybrid': {
        'content_weight': 0.8,  # Weight for content-based similarity (0-1)
        'popularity_weight': 0.2,  # Weight for popularity (1 - content_weight)
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': str(BASE_DIR / 'logs' / 'app.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'recommendation_system': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}

# Create necessary directories
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Add more configuration as needed
