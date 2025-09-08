"""
Optimized TacticalViewConverter with vectorized keypoint validation.

Key improvements:
- O(n²) complexity instead of O(n⁴) 
- Vectorized numpy operations
- Robust statistical methods
- Clear separation of concerns
- Comprehensive error handling
"""
import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple, Union
from copy import deepcopy

from .court_model import CourtModel, DEFAULT_COURT
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.geometry import (
    pairwise_sq_distance, 
    robust_scale_estimate, 
    detect_outliers_by_scale,
    validate_middle_line,
    frames_to_arrays
)

logger = logging.getLogger(__name__)


class TacticalViewConverterOptimized:
    """
    Optimized tactical view converter for basketball court analysis.
    
    Provides efficient keypoint validation and court coordinate transformation
    with O(n²) algorithmic complexity and vectorized operations.
    """
    
    def __init__(self, 
                 court_model: Optional[CourtModel] = None,
                 court_image_path: Optional[str] = None):
        """
        Initialize the tactical view converter.
        
        Args:
            court_model: Court model with dimensions and keypoints
            court_image_path: Path to court image (overrides court_model path if provided)
        """
        self.court_model = court_model or DEFAULT_COURT
        
        # Override court image path if provided
        if court_image_path:
            # Create a new court model with updated path
            self.court_model = CourtModel(
                width=self.court_model.width,
                height=self.court_model.height,
                actual_width_meters=self.court_model.actual_width_meters,
                actual_height_meters=self.court_model.actual_height_meters,
                court_image_path=court_image_path
            )
        
        # Cache frequently used properties
        self._template_keypoints = self.court_model.keypoints
        self._template_distances_sq = self.court_model.template_distances_sq
        self._n_keypoints = self.court_model.n_keypoints
    
    @property
    def court_image_path(self) -> str:
        """Get court image path."""
        return self.court_model.court_image_path
    
    @property 
    def width(self) -> int:
        """Get court width."""
        return self.court_model.width
    
    @property
    def height(self) -> int:
        """Get court height."""
        return self.court_model.height
    
    @property
    def key_points(self) -> List[Tuple[int, int]]:
        """Get keypoints in legacy format for backward compatibility."""
        return [tuple(map(int, kp)) for kp in self._template_keypoints]
    
    def validate_keypoints(self, 
                          keypoints_list: List,
                          distance_threshold: float = 0.25,
                          min_failure_ratio: float = 0.6,
                          min_keypoints: int = 4,
                          middle_line_threshold: float = 0.25,
                          scale_method: str = "median") -> List:
        """
        Validate detected keypoints with optimized O(n²) algorithm.
        
        Args:
            keypoints_list: List of YOLO keypoint objects per frame
            distance_threshold: Maximum allowed scale deviation (default: 25%)
            min_failure_ratio: Min fraction of failed comparisons to mark outlier
            min_keypoints: Minimum keypoints required for validation
            middle_line_threshold: Max deviation ratio for middle line validation
            scale_method: Scale estimation method ("median" or "trimmed_mean")
            
        Returns:
            List of validated keypoint objects (copy of input with invalid points zeroed)
        """
        if not keypoints_list:
            logger.warning("Empty keypoints list provided")
            return keypoints_list
        
        try:
            # Convert to numpy arrays for efficient processing
            frames_array = frames_to_arrays(keypoints_list)
            n_frames, n_keypoints, _ = frames_array.shape
            
            if n_keypoints != self._n_keypoints:
                raise ValueError(f"Expected {self._n_keypoints} keypoints, got {n_keypoints}")
            
            # Create output copy
            result_keypoints = deepcopy(keypoints_list)
            
            # Process each frame
            for frame_idx in range(n_frames):
                points = frames_array[frame_idx]
                
                try:
                    valid_mask = self._validate_frame(
                        points, 
                        distance_threshold,
                        min_failure_ratio, 
                        min_keypoints,
                        middle_line_threshold,
                        scale_method
                    )
                    
                    # Apply validation results to output
                    self._apply_validation_mask(result_keypoints[frame_idx], valid_mask)
                    
                except Exception as e:
                    logger.error(f"Error validating frame {frame_idx}: {e}")
                    # Continue with other frames
                    continue
            
            return result_keypoints
            
        except Exception as e:
            logger.error(f"Critical error in keypoint validation: {e}")
            # Return original data on critical errors
            return deepcopy(keypoints_list)
    
    def _validate_frame(self,
                       points: np.ndarray,
                       distance_threshold: float,
                       min_failure_ratio: float,
                       min_keypoints: int,
                       middle_line_threshold: float,
                       scale_method: str) -> np.ndarray:
        """
        Validate keypoints for a single frame.
        
        Args:
            points: Array of shape (n, 2) with keypoint coordinates (NaN for missing)
            distance_threshold: Maximum allowed scale deviation
            min_failure_ratio: Minimum failure ratio to mark as outlier
            min_keypoints: Minimum keypoints required for validation
            middle_line_threshold: Middle line deviation threshold
            scale_method: Scale estimation method
            
        Returns:
            Boolean mask indicating valid keypoints
        """
        n = len(points)
        
        # Check if we have enough detected keypoints
        detected_mask = ~np.isnan(points[:, 0])
        n_detected = detected_mask.sum()
        
        if n_detected < min_keypoints:
            logger.debug(f"Insufficient keypoints: {n_detected} < {min_keypoints}")
            return np.zeros(n, dtype=bool)
        
        # Compute pairwise squared distances
        detected_distances_sq = pairwise_sq_distance(points)
        
        # Compute distance ratios (NaN values propagate automatically)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.sqrt(detected_distances_sq / self._template_distances_sq)
        
        # Estimate global scale factor
        scale = robust_scale_estimate(ratios, method=scale_method)
        
        if scale <= 0 or not np.isfinite(scale):
            logger.warning(f"Invalid scale factor: {scale}")
            return np.zeros(n, dtype=bool)
        
        # Detect outliers based on scale consistency
        outliers = detect_outliers_by_scale(
            ratios, scale, distance_threshold, min_failure_ratio
        )
        
        # Start with detected keypoints, remove outliers
        valid_mask = detected_mask & ~outliers
        
        # Special validation for middle line (index 7)
        if detected_mask[7] and not outliers[7]:
            middle_line_valid = validate_middle_line(
                points, middle_idx=7, max_deviation_ratio=middle_line_threshold
            )
            if not middle_line_valid:
                valid_mask[7] = False
                logger.debug("Middle line failed position validation")
        
        return valid_mask
    
    def _apply_validation_mask(self, keypoint_obj, valid_mask: np.ndarray):
        """
        Apply validation mask to YOLO keypoint object.
        
        Args:
            keypoint_obj: YOLO keypoint object to modify
            valid_mask: Boolean mask indicating valid keypoints
        """
        try:
            # Zero out invalid keypoints
            invalid_indices = np.where(~valid_mask)[0]
            
            for i in invalid_indices:
                if hasattr(keypoint_obj, 'xy') and hasattr(keypoint_obj, 'xyn'):
                    keypoint_obj.xy[0][i] *= 0
                    keypoint_obj.xyn[0][i] *= 0
                    
        except Exception as e:
            logger.error(f"Error applying validation mask: {e}")
    
    def get_validation_stats(self, 
                           keypoints_list: List,
                           validated_keypoints: List) -> dict:
        """
        Get statistics about keypoint validation results.
        
        Args:
            keypoints_list: Original keypoints
            validated_keypoints: Validated keypoints
            
        Returns:
            Dictionary with validation statistics
        """
        try:
            original_frames = frames_to_arrays(keypoints_list)
            validated_frames = frames_to_arrays(validated_keypoints)
            
            original_detected = (~np.isnan(original_frames[:, :, 0])).sum()
            validated_detected = (~np.isnan(validated_frames[:, :, 0])).sum()
            
            return {
                'total_frames': len(keypoints_list),
                'original_detections': int(original_detected),
                'validated_detections': int(validated_detected),
                'rejected_detections': int(original_detected - validated_detected),
                'rejection_rate': float(original_detected - validated_detected) / max(original_detected, 1)
            }
        except Exception as e:
            logger.error(f"Error computing validation stats: {e}")
            return {}


# Backward compatibility alias
# TacticalViewConverter = TacticalViewConverterOptimized
