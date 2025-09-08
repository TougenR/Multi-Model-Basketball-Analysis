"""
Unit tests for the optimized TacticalViewConverter.

Tests cover geometry utilities, court model, and keypoint validation logic.
"""
import unittest
import numpy as np
from unittest.mock import Mock, patch

from .court_model import CourtModel, DEFAULT_COURT
from .tactical_view_converter_optimized import TacticalViewConverterOptimized
from utils.geometry import (
    pairwise_sq_distance,
    robust_scale_estimate, 
    detect_outliers_by_scale,
    validate_middle_line,
    frames_to_arrays
)


class TestGeometryUtils(unittest.TestCase):
    """Test geometry utility functions."""
    
    def test_pairwise_sq_distance(self):
        """Test pairwise squared distance calculation."""
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        distances = pairwise_sq_distance(points)
        
        expected = np.array([
            [0, 1, 1],
            [1, 0, 2], 
            [1, 2, 0]
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(distances, expected)
    
    def test_pairwise_sq_distance_with_nan(self):
        """Test that NaN values propagate correctly."""
        points = np.array([[0, 0], [np.nan, np.nan], [1, 1]], dtype=np.float32)
        distances = pairwise_sq_distance(points)
        
        # Check that distances involving NaN points are NaN
        self.assertTrue(np.isnan(distances[1, 0]))
        self.assertTrue(np.isnan(distances[0, 1]))
        self.assertTrue(np.isnan(distances[1, 2]))
        
        # Check that valid distances are computed correctly
        np.testing.assert_almost_equal(distances[0, 2], 2.0)
    
    def test_robust_scale_estimate_median(self):
        """Test median scale estimation."""
        ratios = np.array([1.0, 1.1, 0.9, 1.05, np.nan, 1.2])
        scale = robust_scale_estimate(ratios, method="median")
        np.testing.assert_almost_equal(scale, 1.05)
    
    def test_robust_scale_estimate_trimmed_mean(self):
        """Test trimmed mean scale estimation."""
        ratios = np.array([0.5, 1.0, 1.1, 0.9, 1.05, 1.2, 2.0])  # outliers: 0.5, 2.0
        scale = robust_scale_estimate(ratios, method="trimmed_mean")
        
        # Should exclude extreme values and average the middle ones
        expected = np.mean([0.9, 1.0, 1.05, 1.1, 1.2])
        np.testing.assert_almost_equal(scale, expected, decimal=3)
    
    def test_detect_outliers_by_scale(self):
        """Test outlier detection based on scale consistency."""
        # Create ratios where keypoint 1 is consistently off
        ratios = np.array([
            [1.0, 2.0, 1.1, 1.0],  # keypoint 0: good except with keypoint 1
            [2.0, 1.0, 2.1, 2.0],  # keypoint 1: consistently off (outlier)
            [1.1, 2.1, 1.0, 1.1],  # keypoint 2: good except with keypoint 1  
            [1.0, 2.0, 1.1, 1.0]   # keypoint 3: good except with keypoint 1
        ])
        
        scale = 1.0
        outliers = detect_outliers_by_scale(ratios, scale, threshold=0.3, min_failure_ratio=0.5)
        
        # Only keypoint 1 should be detected as outlier
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(outliers, expected)
    
    def test_validate_middle_line_valid(self):
        """Test middle line validation with valid position."""
        points = np.array([
            [0, 0], [100, 0], [200, 0],  # other points spanning 0-200
            [np.nan, np.nan],            # missing point
            [100, 50]                    # middle line at x=100 (center)
        ])
        
        is_valid = validate_middle_line(points, middle_idx=4)
        self.assertTrue(is_valid)
    
    def test_validate_middle_line_invalid(self):
        """Test middle line validation with invalid position."""
        points = np.array([
            [0, 0], [100, 0], [200, 0],  # other points spanning 0-200  
            [np.nan, np.nan],            # missing point
            [50, 50]                     # middle line at x=50 (too far from center)
        ])
        
        is_valid = validate_middle_line(points, middle_idx=4, max_deviation_ratio=0.1)
        self.assertFalse(is_valid)
    
    def test_frames_to_arrays(self):
        """Test conversion from YOLO format to numpy arrays."""
        # Mock YOLO keypoint objects
        mock_frame1 = Mock()
        mock_frame1.xy.tolist.return_value = [[[10, 20], [0, 0], [30, 40]]]
        
        mock_frame2 = Mock()  
        mock_frame2.xy.tolist.return_value = [[[15, 25], [35, 45], [0, 0]]]
        
        frames = [mock_frame1, mock_frame2]
        arrays = frames_to_arrays(frames)
        
        expected = np.array([
            [[10, 20], [np.nan, np.nan], [30, 40]],
            [[15, 25], [35, 45], [np.nan, np.nan]]
        ], dtype=np.float32)
        
        # Check shapes and finite values
        self.assertEqual(arrays.shape, (2, 3, 2))
        np.testing.assert_array_equal(arrays[0, 0], [10, 20])
        np.testing.assert_array_equal(arrays[1, 1], [35, 45])
        
        # Check that (0,0) coordinates became NaN
        self.assertTrue(np.isnan(arrays[0, 1]).all())
        self.assertTrue(np.isnan(arrays[1, 2]).all())


class TestCourtModel(unittest.TestCase):
    """Test CourtModel dataclass."""
    
    def test_default_court_creation(self):
        """Test creation of default court model."""
        court = CourtModel()
        
        self.assertEqual(court.width, 300)
        self.assertEqual(court.height, 161)
        self.assertEqual(court.n_keypoints, 18)
        self.assertIsInstance(court.keypoints, np.ndarray)
        self.assertEqual(court.keypoints.shape, (18, 2))
    
    def test_custom_court_creation(self):
        """Test creation of custom court model."""
        court = CourtModel(width=400, height=200, actual_width_meters=30)
        
        self.assertEqual(court.width, 400)
        self.assertEqual(court.height, 200)
        self.assertEqual(court.actual_width_meters, 30)
    
    def test_template_distances_computation(self):
        """Test that template distances are computed correctly."""
        court = CourtModel()
        distances = court.template_distances_sq
        
        # Should be symmetric matrix
        self.assertEqual(distances.shape, (18, 18))
        np.testing.assert_array_equal(distances, distances.T)
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(distances), np.zeros(18))
    
    def test_scale_factor_calculation(self):
        """Test scale factor calculation."""
        court = CourtModel(width=300, height=150)
        x_scale, y_scale = court.get_scale_factor(600, 300)
        
        self.assertEqual(x_scale, 2.0)
        self.assertEqual(y_scale, 2.0)
    
    def test_keypoint_validation(self):
        """Test keypoint index validation."""
        court = CourtModel()
        
        # Valid indices should not raise
        court.validate_keypoint_indices([0, 5, 17])
        
        # Invalid indices should raise ValueError
        with self.assertRaises(ValueError):
            court.validate_keypoint_indices([18, 19])
        
        with self.assertRaises(ValueError):
            court.validate_keypoint_indices([-1])


class TestTacticalViewConverterOptimized(unittest.TestCase):
    """Test optimized tactical view converter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = TacticalViewConverterOptimized()
    
    def test_initialization(self):
        """Test converter initialization."""
        self.assertIsInstance(self.converter.court_model, CourtModel)
        self.assertEqual(self.converter.width, 300)
        self.assertEqual(self.converter.height, 161)
    
    def test_custom_court_image_path(self):
        """Test converter with custom court image path."""
        custom_path = "/custom/path/court.png"
        converter = TacticalViewConverterOptimized(court_image_path=custom_path)
        
        self.assertEqual(converter.court_image_path, custom_path)
    
    def test_validate_empty_keypoints(self):
        """Test validation with empty keypoint list."""
        result = self.converter.validate_keypoints([])
        self.assertEqual(result, [])
    
    def test_validate_insufficient_keypoints(self):
        """Test validation with insufficient keypoints."""
        # Mock frame with only 2 detected keypoints
        mock_frame = Mock()
        mock_frame.xy.tolist.return_value = [[[10, 20], [0, 0]] + [[0, 0]] * 16]
        mock_frame.xyn = Mock()
        
        # Create deep copy for result  
        result_frame = Mock()
        result_frame.xy = [[list(coord) for coord in mock_frame.xy.tolist()[0]]]
        result_frame.xyn = [[list(coord) for coord in mock_frame.xy.tolist()[0]]]
        
        with patch('tactical_view_converter.tactical_view_converter_optimized.deepcopy', 
                  return_value=[result_frame]):
            result = self.converter.validate_keypoints([mock_frame], min_keypoints=4)
            
            # Should return the input unchanged when insufficient keypoints
            self.assertEqual(len(result), 1)
    
    def test_backward_compatibility_properties(self):
        """Test that legacy properties work correctly."""
        keypoints = self.converter.key_points
        
        self.assertIsInstance(keypoints, list)
        self.assertEqual(len(keypoints), 18)
        self.assertIsInstance(keypoints[0], tuple)
        self.assertEqual(len(keypoints[0]), 2)
    
    def test_validation_stats(self):
        """Test validation statistics computation."""
        # Mock original and validated keypoints
        mock_original = Mock()
        mock_original.xy.tolist.return_value = [[[10, 20], [30, 40], [0, 0]] + [[50, 60]] * 15]
        
        mock_validated = Mock()  
        mock_validated.xy.tolist.return_value = [[[10, 20], [0, 0], [0, 0]] + [[50, 60]] * 15]
        
        stats = self.converter.get_validation_stats([mock_original], [mock_validated])
        
        self.assertIn('total_frames', stats)
        self.assertIn('original_detections', stats)
        self.assertIn('validated_detections', stats)
        self.assertIn('rejection_rate', stats)


if __name__ == '__main__':
    # Set up logging to see debug messages during tests
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()
