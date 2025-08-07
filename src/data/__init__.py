"""
Data module for the Image Recommendation System.

This module handles loading, processing, and managing image data.
"""

from .data_loader import load_image_data, create_mock_dataset  # noqa: F401

__all__ = ['load_image_data', 'create_mock_dataset']
