# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Run tests**: `python3 -m tactical_view_converter.test_optimized`
- **Run performance comparison**: `python3 tactical_view_converter/performance_comparison.py`
- **Test individual components**: Import and test classes directly in Python REPL

## Architecture & Core Components

This module provides tactical view conversion for basketball court analysis with two main implementations:

### Original vs Optimized Implementation

**Original (`tactical_view_converter.py`)**:
- O(n⁴) algorithmic complexity with nested loops
- Simple validation approach with scale consistency checks
- Direct distance calculations between keypoint pairs
- Basic outlier detection for obvious misplacements

**Optimized (`tactical_view_converter_optimized.py`)**:
- O(n²) complexity with vectorized numpy operations
- Comprehensive statistical methods (median, trimmed_mean)
- Pre-computed template distance matrices
- Robust error handling and validation statistics
- Advanced homography transformation with fallback scaling
- Team assignment integration for multi-color visualization

### Key Components

**Court Model (`court_model.py`)**:
- `CourtModel` dataclass with standardized court dimensions
- 18 keypoints: corners, edges, center line, free throw lines
- Pre-computed distance matrices for performance
- Helper methods for keypoint naming and validation

**Geometry Utils (`utils/geometry.py`)**:
- `pairwise_sq_distance()`: Vectorized distance calculations
- `robust_scale_estimate()`: Statistical scale factor estimation
- `detect_outliers_by_scale()`: Scale-based outlier detection
- `validate_middle_line()`: Specialized middle line validation

**Homography (`homography.py`)**:
- `Homography` class for perspective transformation
- Robust validation with degenerate detection
- Point transformation between coordinate systems
- Error computation and validation statistics

**Main Classes**:
- `TacticalViewConverter`: Legacy implementation for backward compatibility
- `TacticalViewConverterOptimized`: High-performance implementation with team support
- Both support same API with enhanced parameters in optimized version

## Code Style & Conventions

### Python Style
- **Type hints**: Full type annotations throughout optimized code
- **Docstrings**: Comprehensive parameter descriptions and return types
- **Error handling**: Structured exception handling with informative messages
- **Logging**: Use `logging` module instead of print statements
- **Immutable data**: Input data never modified directly, use `deepcopy` when needed

### Algorithmic Patterns
- **Vectorization**: Prefer numpy operations over Python loops
- **Pre-computation**: Cache expensive calculations at initialization
- **Squared distances**: Use squared distances to avoid expensive sqrt operations
- **NaN handling**: Proper propagation of invalid/missing data points

### Statistical Methods
- **Robust estimation**: Use median or trimmed_mean instead of simple averages
- **Outlier detection**: Statistical consistency-based rather than threshold-based
- **Scale estimation**: Multi-reference approach for reliability

### Performance Considerations
- **Complexity**: O(n²) vs O(n⁴) makes major difference for 18 keypoints
- **Memory**: Pre-computed matrices (~1.3KB for 18 keypoints)
- **Caching**: Store frequently accessed properties as private attributes
- **Fallback systems**: Multiple layers of transformation fallbacks

## Testing & Validation

### Test Structure
- **Comprehensive unit tests**: 19 test cases covering all major functions
- **Mock data generation**: Controlled testing scenarios
- **Edge cases**: Empty inputs, malformed data, boundary conditions
- **Performance benchmarks**: Comparison between implementations

### Validation Parameters (Optimized Version)
- `distance_threshold`: Maximum allowed scale deviation (default: 0.5 = 50%)
- `min_failure_ratio`: Failure ratio threshold for outlier detection (default: 0.8)
- `min_keypoints`: Minimum keypoints required for validation (default: 4)
- `scale_method`: Statistical method for scale estimation ("median" or "trimmed_mean")

### Keypoint Index Reference
```
0: top-left corner        9: left free throw bottom
1: left edge 1           10: bottom-right corner
2: left edge 2           11: right edge 1
3: left center           12: right center
4: left edge 3           13: right edge 2
5: bottom-left corner    14: right edge 3
6: bottom center         15: top-right corner
7: top center            16: right free throw top
8: left free throw top   17: right free throw bottom
```

## Integration Notes

### Backward Compatibility
- Optimized version maintains same method signatures as original
- Legacy `key_points` property provided for existing code
- Same YOLO format input/output handling
- Team assignment integration requires separate data structure

### Dependencies
- **Core**: numpy, cv2, typing, dataclasses
- **External**: utils.geometry (from parent directory)
- **Optional**: logging for debug output

### Court Dimensions
- **Default**: 400x200 pixels representing 30x15 meters
- **Configurable**: Custom court models supported via `CourtModel`
- **Key points**: 18 standardized positions covering court boundaries and important lines

### Tactical View Features
- **Perspective Transformation**: Full homography-based transformation
- **Fallback Scaling**: Simple scaling when homography fails
- **Team Visualization**: Multi-color team support (red/blue)
- **Player Positioning**: Bounded coordinates within tactical view
- **Court Overlay**: Hardcoded template for consistent layout

### Team Assignment Integration
- **Data Structure**: Team assignments in separate `players_assignment` array
- **Color Mapping**: Team 1 (Red), Team 2 (Blue), with contrasting text colors
- **Position Validation**: Strict bounding to tactical view dimensions
- **Robust Handling**: Default team assignment when data missing

## Usage Examples

### Basic Tactical View with Teams
```python
from tactical_view_converter import TacticalViewConverterOptimized, CourtModel
from tactical_view_drawer import TacticalViewDrawer

# Initialize converter with court model
court = CourtModel(width=400, height=200, actual_width_meters=30)
converter = TacticalViewConverterOptimized(court_model=court)

# Initialize drawer with perspective mode
drawer = TacticalViewDrawer(perspective_mode=True, transparency=0.7)

# Draw with team assignments
output_frames = drawer.draw(
    video_frames,
    court_image_path,
    width, height,
    court_keypoints,
    player_tracks=player_tracks,
    ball_tracks=ball_tracks,
    players_assignment=team_assignments,  # Required for team colors
    converter=converter
)
```

### Direct Player Transformation
```python
# Transform players to tactical view
tactical_players = converter.transform_players_to_tactical_view(
    player_tracks, detected_keypoints, frame_idx, frame_shape
)

# Transform ball position
tactical_ball = converter.transform_ball_to_tactical_view(
    ball_tracks, detected_keypoints, frame_idx, frame_shape
)
```