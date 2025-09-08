"""
Basketball court model with standardized dimensions and keypoint definitions.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CourtModel:
    """
    Basketball court model with standardized dimensions and keypoint layout.
    
    Provides court dimensions, keypoint positions, and related calculations
    in a centralized, immutable structure.
    """
    # Court dimensions in pixels (tactical view)
    width: int = 300
    height: int = 161
    
    # Actual court dimensions in meters
    actual_width_meters: float = 28.0
    actual_height_meters: float = 15.0
    
    # Court image path for visualization
    court_image_path: str = "./images/basketball_court.png"
    
    def __post_init__(self):
        """Initialize computed properties after dataclass creation."""
        self._keypoints = self._compute_keypoints()
        self._template_distances_sq = self._compute_template_distances_squared()
    
    def _compute_keypoints(self) -> np.ndarray:
        """
        Compute standardized court keypoint positions.
        
        Returns:
            Array of shape (n, 2) with keypoint coordinates
        """
        # Helper function for y-coordinate conversion
        def y_coord(meters: float) -> int:
            return int((meters / self.actual_height_meters) * self.height)
        
        # Helper function for x-coordinate conversion  
        def x_coord(meters: float) -> int:
            return int((meters / self.actual_width_meters) * self.width)
        
        keypoints = [
            # Left edge keypoints (x=0)
            (0, 0),                                    # 0: top-left corner
            (0, y_coord(0.91)),                       # 1: left edge point 1
            (0, y_coord(5.18)),                       # 2: left edge point 2
            (0, y_coord(10.0)),                       # 3: left edge center
            (0, y_coord(14.1)),                       # 4: left edge point 3
            (0, self.height),                         # 5: bottom-left corner
            
            # Center line
            (self.width // 2, self.height),           # 6: bottom center
            (self.width // 2, 0),                     # 7: top center (middle line)
            
            # Left free throw line
            (x_coord(5.79), y_coord(5.18)),          # 8: left free throw top
            (x_coord(5.79), y_coord(10.0)),          # 9: left free throw bottom
            
            # Right edge keypoints (x=width)
            (self.width, self.height),                # 10: bottom-right corner
            (self.width, y_coord(14.1)),             # 11: right edge point 1
            (self.width, y_coord(10.0)),             # 12: right edge center
            (self.width, y_coord(5.18)),             # 13: right edge point 2
            (self.width, y_coord(0.91)),             # 14: right edge point 3
            (self.width, 0),                         # 15: top-right corner
            
            # Right free throw line
            (x_coord(self.actual_width_meters - 5.79), y_coord(5.18)),  # 16: right free throw top
            (x_coord(self.actual_width_meters - 5.79), y_coord(10.0)),  # 17: right free throw bottom
        ]
        
        return np.array(keypoints, dtype=np.float32)
    
    def _compute_template_distances_squared(self) -> np.ndarray:
        """
        Pre-compute squared distances between all template keypoint pairs.
        
        Returns:
            Matrix of shape (n, n) with squared distances
        """
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.geometry import pairwise_sq_distance
        return pairwise_sq_distance(self._keypoints)
    
    @property
    def keypoints(self) -> np.ndarray:
        """Get template keypoint positions."""
        return self._keypoints.copy()
    
    @property
    def template_distances_sq(self) -> np.ndarray:
        """Get pre-computed squared distances between template keypoints."""
        return self._template_distances_sq.copy()
    
    @property
    def n_keypoints(self) -> int:
        """Get number of keypoints."""
        return len(self._keypoints)
    
    def get_scale_factor(self, pixel_width: float, pixel_height: float) -> Tuple[float, float]:
        """
        Calculate scale factors between template and actual court dimensions.
        
        Args:
            pixel_width: Detected court width in pixels
            pixel_height: Detected court height in pixels
            
        Returns:
            Tuple of (x_scale, y_scale) factors
        """
        x_scale = pixel_width / self.width
        y_scale = pixel_height / self.height
        return x_scale, y_scale
    
    def validate_keypoint_indices(self, indices: List[int]) -> None:
        """
        Validate that keypoint indices are within valid range.
        
        Args:
            indices: List of keypoint indices to validate
            
        Raises:
            ValueError: If any index is out of range
        """
        invalid = [i for i in indices if i < 0 or i >= self.n_keypoints]
        if invalid:
            raise ValueError(f"Invalid keypoint indices: {invalid}. Valid range: 0-{self.n_keypoints-1}")
    
    def get_keypoint_name(self, index: int) -> str:
        """
        Get descriptive name for keypoint index.
        
        Args:
            index: Keypoint index
            
        Returns:
            Human-readable keypoint name
        """
        names = [
            "top-left corner", "left edge 1", "left edge 2", "left center", 
            "left edge 3", "bottom-left corner", "bottom center", "top center",
            "left free throw top", "left free throw bottom", "bottom-right corner",
            "right edge 1", "right center", "right edge 2", "right edge 3",
            "top-right corner", "right free throw top", "right free throw bottom"
        ]
        
        if 0 <= index < len(names):
            return names[index]
        return f"keypoint_{index}"


# Default court model instance
DEFAULT_COURT = CourtModel()
