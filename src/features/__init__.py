"""
Feature extraction module for the Image Recommendation System.

This module handles extracting features from images using various techniques
including pre-trained CNNs, color histograms, and texture analysis.
"""

from .feature_extractor import (
    FeatureExtractor,
    extract_features_batch,
    get_feature_extractor
)

__all__ = [
    'FeatureExtractor',
    'extract_features_batch',
    'get_feature_extractor'
]
