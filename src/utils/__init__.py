"""
Utility functions for the Image Recommendation System.

This module contains helper functions and utilities used throughout the project.
"""

from .utils import (
    create_directory,
    get_file_extension,
    is_image_file,
    get_image_files,
    load_image,
    save_image,
    get_image_metadata
)

__all__ = [
    'create_directory',
    'get_file_extension',
    'is_image_file',
    'get_image_files',
    'load_image',
    'save_image',
    'get_image_metadata',
    'display_images'
]
