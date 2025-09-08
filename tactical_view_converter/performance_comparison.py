"""
Performance comparison between original and optimized TacticalViewConverter.
"""
import time
import numpy as np
from unittest.mock import Mock

try:
    from .tactical_view_converter import TacticalViewConverter as OriginalConverter
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    print("Original converter not available for comparison")

from .tactical_view_converter_optimized import TacticalViewConverterOptimized as OptimizedConverter


def create_mock_keypoints(n_frames: int = 100, n_keypoints: int = 18, detection_rate: float = 0.7):
    """Create mock keypoint data for testing."""
    frames = []
    
    for _ in range(n_frames):
        mock_frame = Mock()
        
        # Generate random keypoints with some missing (0,0) 
        keypoints = []
        for i in range(n_keypoints):
            if np.random.random() < detection_rate:
                # Add some noise to make it realistic
                base_x = (i % 6) * 50 + np.random.normal(0, 5)
                base_y = (i // 6) * 30 + np.random.normal(0, 5)
                keypoints.append([max(0, base_x), max(0, base_y)])
            else:
                keypoints.append([0, 0])  # Missing detection
        
        mock_frame.xy.tolist.return_value = [keypoints]
        mock_frame.xyn = Mock()
        frames.append(mock_frame)
    
    return frames


def benchmark_converter(converter, keypoints_data, n_runs: int = 5):
    """Benchmark a converter with the given data."""
    times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        result = converter.validate_keypoints(keypoints_data)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def main():
    """Run performance comparison."""
    print("Basketball Court Keypoint Validation Performance Comparison")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {"n_frames": 50, "detection_rate": 0.8, "name": "Small (50 frames)"},
        {"n_frames": 200, "detection_rate": 0.7, "name": "Medium (200 frames)"},  
        {"n_frames": 500, "detection_rate": 0.6, "name": "Large (500 frames)"},
    ]
    
    for config in test_configs:
        print(f"\n{config['name']}:")
        print("-" * 40)
        
        # Generate test data
        keypoints_data = create_mock_keypoints(
            n_frames=config['n_frames'],
            detection_rate=config['detection_rate']
        )
        
        # Test optimized version
        optimized_converter = OptimizedConverter()
        opt_results = benchmark_converter(optimized_converter, keypoints_data)
        
        print(f"Optimized Converter:")
        print(f"  Mean time: {opt_results['mean_time']*1000:.2f} ms")
        print(f"  Std time:  {opt_results['std_time']*1000:.2f} ms")
        print(f"  Min time:  {opt_results['min_time']*1000:.2f} ms")
        print(f"  Max time:  {opt_results['max_time']*1000:.2f} ms")
        
        # Test original version if available
        if ORIGINAL_AVAILABLE:
            original_converter = OriginalConverter()
            orig_results = benchmark_converter(original_converter, keypoints_data)
            
            print(f"\nOriginal Converter:")
            print(f"  Mean time: {orig_results['mean_time']*1000:.2f} ms") 
            print(f"  Std time:  {orig_results['std_time']*1000:.2f} ms")
            print(f"  Min time:  {orig_results['min_time']*1000:.2f} ms")
            print(f"  Max time:  {orig_results['max_time']*1000:.2f} ms")
            
            speedup = orig_results['mean_time'] / opt_results['mean_time']
            print(f"\nSpeedup: {speedup:.1f}x faster")
        
        # Memory and complexity analysis
        n_keypoints = 18
        original_ops = config['n_frames'] * (n_keypoints ** 4)  # O(n^4) per frame
        optimized_ops = config['n_frames'] * (n_keypoints ** 2)  # O(n^2) per frame
        
        theoretical_speedup = original_ops / optimized_ops
        print(f"Theoretical complexity improvement: {theoretical_speedup:.0f}x")
    
    print(f"\n{'='*60}")
    print("Key Improvements:")
    print("• Algorithmic complexity: O(n⁴) → O(n²)")
    print("• Vectorized numpy operations instead of Python loops")  
    print("• Robust statistical methods for scale estimation")
    print("• Better error handling and edge case management")
    print("• Comprehensive unit test coverage")


if __name__ == "__main__":
    main()
