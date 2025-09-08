"""
Optimized geometry utilities for basketball court analysis.
Provides vectorized operations for distance calculations and keypoint validation.
"""
import numpy as np
from typing import Tuple, Optional


def pairwise_sq_distance(points: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared distances between points efficiently.
    
    Args:
        points: Array of shape (n, 2) containing 2D points
        
    Returns:
        Array of shape (n, n) with squared distances between all point pairs
        NaN values are properly propagated for missing points
    """
    if points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (n, 2), got {points.shape}")
    
    # Use broadcasting to compute all pairwise differences
    diff = points[:, None, :] - points[None, :, :]
    return (diff ** 2).sum(axis=-1)


def robust_scale_estimate(ratios: np.ndarray, method: str = "median") -> float:
    """
    Estimate global scale factor robustly from distance ratios.
    
    Args:
        ratios: Array of distance ratios (may contain NaN for invalid pairs)
        method: Estimation method ("median", "trimmed_mean")
        
    Returns:
        Robust scale estimate
    """
    valid_ratios = ratios[np.isfinite(ratios)]
    
    if len(valid_ratios) == 0:
        return 1.0
    
    if method == "median":
        return np.median(valid_ratios)
    elif method == "trimmed_mean":
        # Remove extreme 10% from both ends
        q10, q90 = np.percentile(valid_ratios, [10, 90])
        trimmed = valid_ratios[(valid_ratios >= q10) & (valid_ratios <= q90)]
        return np.mean(trimmed) if len(trimmed) > 0 else np.median(valid_ratios)
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_outliers_by_scale(ratios: np.ndarray, 
                           scale: float,
                           threshold: float = 0.25,
                           min_failure_ratio: float = 0.6) -> np.ndarray:
    """
    Detect outlier keypoints based on scale consistency.
    
    Args:
        ratios: Distance ratios matrix (n, n)
        scale: Expected global scale factor
        threshold: Maximum allowed relative error
        min_failure_ratio: Minimum fraction of failed comparisons to mark as outlier
        
    Returns:
        Boolean array indicating which keypoints are outliers
    """
    n = ratios.shape[0]
    
    # Compute relative errors
    abs_error = np.abs(ratios - scale) / scale
    
    # Count failures per keypoint (excluding NaN comparisons)
    failures = (abs_error > threshold) & np.isfinite(ratios)
    valid_comparisons = np.isfinite(ratios)
    
    outliers = np.zeros(n, dtype=bool)
    
    for i in range(n):
        total_valid = valid_comparisons[i].sum() - 1  # Exclude self-comparison
        if total_valid > 0:
            failure_count = failures[i].sum()
            failure_ratio = failure_count / total_valid
            outliers[i] = failure_ratio > min_failure_ratio
    
    return outliers


def validate_middle_line(points: np.ndarray, 
                        middle_idx: int = 7,
                        max_deviation_ratio: float = 0.25) -> bool:
    """
    Validate middle line position relative to other court features.
    
    Args:
        points: Detected keypoints (n, 2) with NaN for missing points
        middle_idx: Index of middle line keypoint
        max_deviation_ratio: Maximum allowed deviation as fraction of court width
        
    Returns:
        True if middle line position is valid
    """
    # Check if middle line is detected
    if np.isnan(points[middle_idx, 0]):
        return True  # Can't validate if not detected
    
    # Get x-coordinates of other detected points
    mask = ~np.isnan(points[:, 0]) & (np.arange(len(points)) != middle_idx)
    other_x = points[mask, 0]
    
    if len(other_x) < 2:
        return True  # Insufficient data for validation
    
    # Estimate court center and width
    court_center = np.mean(other_x)
    court_width = np.ptp(other_x)  # Peak-to-peak (max - min)
    
    # Check deviation from expected center
    deviation = abs(points[middle_idx, 0] - court_center)
    return deviation <= court_width * max_deviation_ratio


def frames_to_arrays(keypoints_list: list, 
                    default_value: float = np.nan) -> np.ndarray:
    """
    Convert YOLO keypoint format to numpy arrays with NaN for missing points.
    
    Args:
        keypoints_list: List of YOLO keypoint objects per frame
        default_value: Value to use for missing/invalid keypoints
        
    Returns:
        Array of shape (frames, n_keypoints, 2)
    """
    frames = []
    
    for frame_keypoints in keypoints_list:
        if hasattr(frame_keypoints, 'xy'):
            # YOLO format
            points = frame_keypoints.xy.tolist()[0]
        else:
            # Assume it's already a list of [x, y] coordinates
            points = frame_keypoints
        
        # Convert to numpy and replace invalid coordinates
        points_array = np.array(points, dtype=np.float32)
        
        # Replace (0, 0) coordinates with NaN (common for missing detections)
        invalid_mask = (points_array[:, 0] <= 0) | (points_array[:, 1] <= 0)
        points_array[invalid_mask] = default_value
        
        frames.append(points_array)
    
    return np.array(frames)
