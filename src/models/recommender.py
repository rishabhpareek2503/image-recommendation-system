"""
Recommendation engine for the Image Recommendation System.

This module implements content-based and hybrid recommendation algorithms
for suggesting similar images based on visual features.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class ContentBasedRecommender:
    """A content-based recommender system for images."""
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """Initialize the recommender with optional feature weights.
        
        Args:
            feature_weights: Dictionary mapping feature names to weights.
                           If None, equal weights will be used.
        """
        self.feature_weights = feature_weights or {
            'cnn_features': 0.7,
            'color_histograms': 0.3
        }
        self.scaler = MinMaxScaler()
    
    def _normalize_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize feature vectors to the same scale.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Dictionary of normalized feature arrays
        """
        normalized = {}
        for name, feat in features.items():
            # Make a copy to avoid modifying the original
            feat = feat.copy()
            
            # Reshape to 2D array if needed
            if len(feat.shape) == 1:
                feat = feat.reshape(1, -1)
                
            # Standardize features (zero mean, unit variance)
            mean = np.mean(feat, axis=1, keepdims=True)
            std = np.std(feat, axis=1, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            
            feat = (feat - mean) / std
            
            # Clip extreme values
            feat = np.clip(feat, -3, 3)
            
            # Min-max scale to [0, 1]
            min_vals = np.min(feat, axis=1, keepdims=True)
            max_vals = np.max(feat, axis=1, keepdims=True)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            
            feat = (feat - min_vals) / range_vals
            
            normalized[name] = feat
            
        return normalized
    
    def calculate_similarity(self, query_features: Dict[str, np.ndarray], 
                           target_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate similarity between query and target features.
        
        Args:
            query_features: Dictionary of feature arrays for the query image(s)
            target_features: Dictionary of feature arrays for the target images
            
        Returns:
            Array of similarity scores for each target image
        """
        # Calculate similarity for each feature type
        similarities = {}
        
        for name in query_features.keys():
            if name in self.feature_weights and name in target_features:
                query_feat = query_features[name]
                target_feat = target_features[name]
                
                # Ensure shapes are correct for matrix multiplication
                if len(query_feat.shape) == 1:
                    query_feat = query_feat.reshape(1, -1)
                if len(target_feat.shape) == 1:
                    target_feat = target_feat.reshape(1, -1)
                
                # Calculate cosine similarity
                query_norm = np.linalg.norm(query_feat, axis=1, keepdims=True)
                target_norm = np.linalg.norm(target_feat, axis=1, keepdims=True)
                
                # Avoid division by zero
                query_norm[query_norm == 0] = 1e-10
                target_norm[target_norm == 0] = 1e-10
                
                # Normalize features
                query_norm_feat = query_feat / query_norm
                target_norm_feat = target_feat / target_norm
                
                # Calculate cosine similarity
                sim = np.dot(target_norm_feat, query_norm_feat.T).flatten()
                
                # If multiple query images, take the maximum similarity for each target
                if len(sim.shape) > 1 and sim.shape[0] > 1:
                    sim = np.max(sim, axis=0)
                
                # Scale to [0, 1] range
                sim = (sim + 1) / 2  # Convert from [-1, 1] to [0, 1]
                
                similarities[name] = sim
        
        # Combine similarities using weighted average
        if not similarities:
            raise ValueError("No valid features found for similarity calculation")
        
        # Initialize combined similarity with zeros
        combined_sim = np.zeros_like(next(iter(similarities.values())))
        total_weight = 0.0
        
        # Add weighted similarities
        for name, sim in similarities.items():
            weight = self.feature_weights.get(name, 0.0)
            if weight > 0:
                combined_sim += sim * weight
                total_weight += weight
        
        # Normalize by total weight (in case weights don't sum to 1)
        if total_weight > 0:
            combined_sim /= total_weight
        
        # Ensure scores are in [0, 1] range
        combined_sim = np.clip(combined_sim, 0, 1)
        
        return combined_sim
    
    def recommend(self, query_features: Dict[str, np.ndarray], 
                 target_features: Dict[str, np.ndarray], 
                 k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend top-k similar images.
        
        Args:
            query_features: Dictionary of feature arrays for the query image(s)
            target_features: Dictionary of feature arrays for the target images
            k: Number of recommendations to return
            
        Returns:
            Tuple of (indices, scores) for the top-k recommendations
        """
        # Calculate similarity scores
        scores = self.calculate_similarity(query_features, target_features)
        
        # Get top-k indices (highest similarity first)
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices, top_k_scores


class HybridRecommender(ContentBasedRecommender):
    """A hybrid recommender that combines content-based and popularity-based recommendations."""
    
    def __init__(self, content_weight: float = 0.8, **kwargs):
        """Initialize the hybrid recommender.
        
        Args:
            content_weight: Weight for content-based similarity (0-1)
            **kwargs: Additional arguments for the base ContentBasedRecommender
        """
        super().__init__(**kwargs)
        self.content_weight = max(0.0, min(1.0, content_weight))
    
    def recommend(self, query_features: Dict[str, np.ndarray], 
                 target_features: Dict[str, np.ndarray], 
                 k: int = 5,
                 popularity_scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend top-k similar images using a hybrid approach.
        
        Args:
            query_features: Dictionary of feature arrays for the query image(s)
            target_features: Dictionary of feature arrays for the target images
            k: Number of recommendations to return
            popularity_scores: Optional array of popularity scores for each target image
            
        Returns:
            Tuple of (indices, scores) for the top-k recommendations
        """
        # Calculate content-based similarity
        content_scores = self.calculate_similarity(query_features, target_features)
        
        # If no popularity scores, fall back to content-based only
        if popularity_scores is None or len(popularity_scores) != len(content_scores):
            return super().recommend(query_features, target_features, k)
        
        # Normalize popularity scores to [0, 1]
        pop_scores = np.array(popularity_scores, dtype=float)
        if np.max(pop_scores) > np.min(pop_scores):
            pop_scores = (pop_scores - np.min(pop_scores)) / (np.max(pop_scores) - np.min(pop_scores))
        
        # Combine scores
        combined_scores = (self.content_weight * content_scores + 
                          (1 - self.content_weight) * pop_scores)
        
        # Get top-k indices (highest combined score first)
        top_k_indices = np.argsort(combined_scores)[::-1][:k]
        top_k_scores = combined_scores[top_k_indices]
        
        return top_k_indices, top_k_scores


def get_recommender(method: str = 'content_based', **kwargs):
    """Factory function to get a recommender instance.
    
    Args:
        method: Type of recommender ('content_based' or 'hybrid')
        **kwargs: Additional arguments for the recommender constructor
        
    Returns:
        Recommender instance
    """
    if method.lower() == 'hybrid':
        return HybridRecommender(**kwargs)
    return ContentBasedRecommender(**kwargs)
