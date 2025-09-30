# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Run main analysis**: `python3 main.py`
- **Run tests**: `python3 test.py` (basic test file, not a formal test framework)
- **Run tactical view converter tests**: `python3 -m tactical_view_converter.test_optimized`
- **Run performance comparison**: `python3 tactical_view_converter/performance_comparison.py`

## Architecture & Core Pipeline

This is a computer vision basketball analysis system that processes video input to track players, balls, and generate tactical visualizations.

### Main Processing Flow
1. **Video Input**: Reads basketball game video from `input_videos/` directory
2. **Object Detection**: YOLO-based models detect players and balls (`PlayerTracker`, `BallTracker`)
3. **Team Assignment**: CLIP-based model assigns players to teams (`TeamAssigner`)
4. **Ball Possession**: Tracks which player controls the ball (`BallAcquisitionDetector`)
5. **Play Analysis**: Detects passes and interceptions (`PassAndInterceptionDetector`)
6. **Court Mapping**: Maps detected keypoints to court model (`CourtKeypointDetector`, `TacticalViewConverter`)
7. **Visualization**: Generates annotated video output with tactical overlays

### Key Components

**Tracker Modules** (`tracker/`)
- `PlayerTracker`: YOLO-based player detection and tracking
- `BallTracker`: Ball detection with position validation and interpolation

**Analysis Modules**
- `team_assigner/`: Team classification using CLIP embeddings
- `ball_acquisition/`: Ball possession detection
- `pass_and_interception/`: Pass and interception event detection
- `court_keypoint_detector/`: Court boundary and feature detection

**Tactical View Conversion** (`tactical_view_converter/`)
- Original `TacticalViewConverter` and optimized `TacticalViewConverterOptimized`
- `CourtModel`: Defines basketball court dimensions and keypoint relationships
- Optimized version offers O(n²) performance vs original O(n⁴) complexity

**Visualization** (`drawers/`)
- Modular drawer classes for different visualization layers
- Separate drawers for players, balls, team control, passes/interceptions, court keypoints, and tactical overlay

### Data Management

**Models**: Pre-trained YOLO models in `models/` (.pt files)
- `player_detector.pt`: Player detection
- `ball_detector_model.pt`: Ball detection
- `court_keypoint_detector.pt`: Court feature detection

**Stubs**: Cached intermediate results in `stubs/` (.pkl files) for faster development iterations
- Player tracks, ball tracks, team assignments, and court keypoints can be cached

**Output**: Annotated video saved to `output_videos/output_videos.avi`

### Development Notes

- Uses stub files to cache expensive computations during development
- All tracker classes support `read_from_stub=True` parameter
- Court dimensions configurable in `CourtModel` (default: 400x200px representing 30m actual width)
- Optimized tactical view converter maintains backward compatibility with original API
- Video processing uses OpenCV; YOLO models via ultralytics; tracking via supervision library