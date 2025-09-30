"""
Keypoint mapping and correspondence utilities for tactical view converter.

Handles the mapping between detected YOLO keypoints and template keypoints,
including support for different camera orientations.
"""
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# YOLO keypoint indices to CourtModel template indices
# Based on the training data labeling scheme
DETECTED_TO_TEMPLATE = {
    0: 0,    # top-left corner
    1: 1,    # left edge 1
    2: 2,    # left edge 2  
    3: 3,    # left center
    4: 4,    # left edge 3
    5: 5,    # bottom-left corner
    6: 6,    # bottom center
    7: 7,    # top center (middle line)
    8: 8,    # left free throw top
    9: 9,    # left free throw bottom
    10: 10,  # bottom-right corner
    11: 11,  # right edge 1
    12: 12,  # right center
    13: 13,  # right edge 2
    14: 14,  # right edge 3
    15: 15,  # top-right corner
    16: 16,  # right free throw top
    17: 17,  # right free throw bottom
}

@dataclass
class OrientationResult:
    """Result of orientation detection."""
    orientation: str  # 'normal', 'mirror_x', 'mirror_y', 'rotate_180'
    template_keypoints: np.ndarray
    reprojection_error: float
    num_inliers: int

def remap_detected_keypoints(detected_keypoints: np.ndarray, 
                           n_template_points: int = 18) -> np.ndarray:
    """
    Remap detected keypoints to template keypoint indices.
    
    Args:
        detected_keypoints: Array of detected keypoints shape (n, 2)
        n_template_points: Number of template keypoints (default: 18)
        
    Returns:
        Array of shape (n_template_points, 2) with NaN for missing keypoints
    """
    # Initialize with NaN for all template positions
    remapped = np.full((n_template_points, 2), np.nan, dtype=np.float32)
    
    # Map detected keypoints to their template positions
    for detected_idx, template_idx in DETECTED_TO_TEMPLATE.items():
        if detected_idx < len(detected_keypoints) and template_idx < n_template_points:
            # Check if detected keypoint is valid (not zero or very close to zero)
            detected_point = detected_keypoints[detected_idx]
            if (not np.isnan(detected_point[0]) and not np.isnan(detected_point[1]) and 
                abs(detected_point[0]) > 1 and abs(detected_point[1]) > 1):
                remapped[template_idx] = detected_point
    
    return remapped

def flip_template_orientation(template_keypoints: np.ndarray, 
                            width: int, height: int, 
                            orientation: str) -> np.ndarray:
    """
    Generate template keypoints for different camera orientations.
    
    Args:
        template_keypoints: Original template keypoints shape (n, 2)
        width: Court width in pixels
        height: Court height in pixels
        orientation: 'normal', 'mirror_x', 'mirror_y', 'rotate_180'
        
    Returns:
        Modified template keypoints for the specified orientation
    """
    flipped = template_keypoints.copy()
    
    if orientation == 'mirror_x':
        # Flip horizontally (left/right sides swap)
        flipped[:, 0] = width - flipped[:, 0]
    elif orientation == 'mirror_y':
        # Flip vertically (top/bottom swap)
        flipped[:, 1] = height - flipped[:, 1]
    elif orientation == 'rotate_180':
        # Rotate 180 degrees (both flips)
        flipped[:, 0] = width - flipped[:, 0]
        flipped[:, 1] = height - flipped[:, 1]
    # 'normal' requires no changes
    
    return flipped

def detect_court_orientation(detected_keypoints: np.ndarray,
                           template_keypoints: np.ndarray,
                           court_width: int,
                           court_height: int,
                           error_threshold: float = 12.0) -> OrientationResult:
    """
    Detect the best court orientation by testing different template orientations.
    
    Args:
        detected_keypoints: Array of detected keypoints shape (n, 2)
        template_keypoints: Template keypoints shape (n, 2)
        court_width: Court width in pixels
        court_height: Court height in pixels
        error_threshold: Maximum acceptable reprojection error
        
    Returns:
        OrientationResult with best orientation and error metrics
    """
    orientations = ['normal', 'mirror_x', 'mirror_y', 'rotate_180']
    best_result = None
    
    for orientation in orientations:
        try:
            # Generate template for this orientation
            oriented_template = flip_template_orientation(
                template_keypoints, court_width, court_height, orientation
            )
            
            # Find valid correspondence pairs
            valid_mask = (~np.isnan(detected_keypoints[:, 0]) & 
                         ~np.isnan(detected_keypoints[:, 1]) &
                         ~np.isnan(oriented_template[:, 0]) & 
                         ~np.isnan(oriented_template[:, 1]))
            
            if valid_mask.sum() < 4:
                continue  # Need at least 4 points for homography
            
            valid_detected = detected_keypoints[valid_mask]
            valid_template = oriented_template[valid_mask]
            
            # Try to compute homography
            from .homography import Homography
            
            homography = Homography(valid_detected, valid_template)
            is_valid, error = homography.validate_transformation(
                valid_detected, valid_template, max_error=error_threshold
            )
            
            if is_valid:
                inliers = homography.get_inliers_mask()
                num_inliers = inliers.sum() if inliers is not None else len(valid_detected)
                
                result = OrientationResult(
                    orientation=orientation,
                    template_keypoints=oriented_template,
                    reprojection_error=error,
                    num_inliers=num_inliers
                )
                
                # Keep the result with lowest error
                if best_result is None or error < best_result.reprojection_error:
                    best_result = result
                    
        except Exception as e:
            logger.debug(f"Failed to test orientation {orientation}: {e}")
            continue
    
    if best_result is None:
        # Fallback to normal orientation
        best_result = OrientationResult(
            orientation='normal',
            template_keypoints=template_keypoints,
            reprojection_error=float('inf'),
            num_inliers=0
        )
        logger.warning("No valid orientation found, using normal as fallback")
    else:
        logger.debug(f"Best orientation: {best_result.orientation} "
                    f"(error: {best_result.reprojection_error:.2f}, "
                    f"inliers: {best_result.num_inliers})")
    
    return best_result

def get_correspondence_pairs(detected_keypoints: np.ndarray,
                           template_keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get valid correspondence pairs between detected and template keypoints.
    
    Args:
        detected_keypoints: Array of detected keypoints shape (n, 2)
        template_keypoints: Array of template keypoints shape (n, 2)
        
    Returns:
        Tuple of (valid_detected, valid_template) arrays with matching pairs
    """
    # Find points that are valid in both arrays
    valid_mask = (~np.isnan(detected_keypoints[:, 0]) & 
                 ~np.isnan(detected_keypoints[:, 1]) &
                 ~np.isnan(template_keypoints[:, 0]) & 
                 ~np.isnan(template_keypoints[:, 1]) &
                 (detected_keypoints[:, 0] > 1) & 
                 (detected_keypoints[:, 1] > 1))
    
    valid_detected = detected_keypoints[valid_mask]
    valid_template = template_keypoints[valid_mask]
    
    logger.debug(f"Found {len(valid_detected)} valid correspondence pairs "
                f"out of {len(detected_keypoints)} total keypoints")
    
    return valid_detected, valid_template
