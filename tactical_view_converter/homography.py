import numpy as np
import cv2
from typing import Optional, Tuple


class Homography:
    """
    Homography transformation for mapping points between source and target coordinate systems.

    Computes the perspective transformation matrix that maps points from a source
    coordinate system to a target coordinate system using OpenCV's findHomography.
    """

    def __init__(self, src: np.ndarray, target: np.ndarray, method: int = cv2.RANSAC,
                 ransac_reproj_threshold: float = 5.0, max_iters: int = 2000,
                 confidence: float = 0.995):
        """
        Initialize homography transformation.

        Args:
            src: Source points array of shape (n, 2)
            target: Target points array of shape (n, 2)
            method: Homography computation method (default: cv2.RANSAC)
            ransac_reproj_threshold: RANSAC reprojection threshold in pixels
            max_iters: Maximum number of RANSAC iterations
            confidence: Confidence level for RANSAC

        Raises:
            ValueError: If source and target arrays have incompatible shapes
        """
        if src.shape != target.shape:
            raise ValueError(f"Source and target must have the same shape. Got {src.shape} and {target.shape}")
        if src.shape[1] != 2:
            raise ValueError(f"Source and target must be 2D points. Got shape with {src.shape[1]} dimensions")
        if len(src) < 4:
            raise ValueError("At least 4 point correspondences are required for homography computation")

        # Convert to float32 for OpenCV
        src = src.astype(np.float32)
        target = target.astype(np.float32)

        # Compute homography matrix
        self.m, self.mask = cv2.findHomography(
            src, target, method, ransac_reproj_threshold,
            maxIters=max_iters, confidence=confidence
        )

        if self.m is None:
            raise ValueError("Failed to compute homography matrix")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from source to target coordinate system.

        Args:
            points: Points to transform, shape (n, 2)

        Returns:
            Transformed points, shape (n, 2)
        """
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError(f"Points must be 2D. Got shape {points.shape}")

        # Use OpenCV's perspective transform
        points_reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_reshaped, self.m)

        return transformed.reshape(-1, 2)

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single point from source to target coordinate system.

        Args:
            point: Point to transform as (x, y)

        Returns:
            Transformed point as (x, y)
        """
        point_array = np.array([point], dtype=np.float32)
        transformed = self.transform_points(point_array)
        return tuple(transformed[0])

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from target back to source coordinate system.

        Args:
            points: Points in target coordinates, shape (n, 2)

        Returns:
            Points in source coordinates, shape (n, 2)
        """
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError(f"Points must be 2D. Got shape {points.shape}")

        # Compute inverse homography matrix
        m_inv = np.linalg.inv(self.m)

        # Use OpenCV's perspective transform with inverse matrix
        points_reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_reshaped, m_inv)

        return transformed.reshape(-1, 2)

    def inverse_transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single point from target back to source coordinate system.

        Args:
            point: Point in target coordinates as (x, y)

        Returns:
            Point in source coordinates as (x, y)
        """
        point_array = np.array([point], dtype=np.float32)
        transformed = self.inverse_transform_points(point_array)
        return tuple(transformed[0])

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Get the 3x3 homography transformation matrix.

        Returns:
            3x3 transformation matrix
        """
        return self.m.copy()

    def get_inliers_mask(self) -> Optional[np.ndarray]:
        """
        Get the mask indicating which point correspondences were inliers.

        Returns:
            Boolean array indicating inliers, or None if not available
        """
        return self.mask.copy() if self.mask is not None else None

    def compute_reprojection_error(self, src: np.ndarray, target: np.ndarray) -> float:
        """
        Compute mean reprojection error for given point correspondences.

        Args:
            src: Source points, shape (n, 2)
            target: Target points, shape (n, 2)

        Returns:
            Mean reprojection error in pixels
        """
        # Transform source points
        transformed_src = self.transform_points(src)

        # Compute Euclidean distances
        errors = np.sqrt(np.sum((transformed_src - target) ** 2, axis=1))

        return float(np.mean(errors))

    def validate_transformation(self, src: np.ndarray, target: np.ndarray,
                              max_error: float = 10.0) -> Tuple[bool, float]:
        """
        Validate the homography transformation by checking reprojection error.

        Args:
            src: Source points for validation, shape (n, 2)
            target: Target points for validation, shape (n, 2)
            max_error: Maximum acceptable mean reprojection error

        Returns:
            Tuple of (is_valid, mean_error) where is_valid is True if error < max_error
        """
        # Check for degenerate homography matrix
        if not self.is_homography_valid():
            return False, float('inf')
            
        mean_error = self.compute_reprojection_error(src, target)
        is_valid = mean_error < max_error

        return is_valid, mean_error
    
    def is_homography_valid(self) -> bool:
        """
        Check if the homography matrix is valid (not degenerate).
        
        Returns:
            True if homography is valid, False if degenerate
        """
        try:
            # Check matrix determinant - should not be zero or very close to zero
            det = np.linalg.det(self.m)
            if abs(det) < 1e-10:
                return False
            
            # Check condition number - high values indicate ill-conditioned matrix
            cond = np.linalg.cond(self.m)
            if cond > 1e12:  # Very high condition number
                return False
                
            # Check if matrix contains NaN or infinite values
            if not np.all(np.isfinite(self.m)):
                return False
                
            # Test transformation of corner points to detect degenerate cases
            test_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
            try:
                transformed = self.transform_points(test_points)
                
                # Check if all points collapse to a line (degenerate case)
                if len(transformed) >= 3:
                    # Calculate area of transformed quadrilateral
                    # If area is very small, transformation is degenerate
                    area = self._calculate_polygon_area(transformed)
                    if area < 100:  # Very small area threshold
                        return False
                        
                # Check if transformation spreads points reasonably
                x_range = transformed[:, 0].max() - transformed[:, 0].min()
                y_range = transformed[:, 1].max() - transformed[:, 1].min()
                if x_range < 10 or y_range < 10:  # Points too close together
                    return False
                    
            except Exception:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula."""
        if len(points) < 3:
            return 0.0
        try:
            x = points[:, 0]
            y = points[:, 1]
            return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
        except Exception:
            return 0.0
