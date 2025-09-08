# Basketball Analysis Codebase Guide

## Build/Run Commands
- **Run main analysis**: `python3 main.py`
- **Run tests**: `python3 test.py` (basic test file, no formal test framework)
- **Check dependencies**: Requires ultralytics, opencv-python, supervision, transformers, pandas, numpy, PIL

## Architecture & Structure
- **Core pipeline**: Computer vision basketball analysis with object tracking, team assignment, and tactical visualization
- **Main modules**: tracker (YOLO-based player/ball detection), team_assigner (CLIP-based team classification), court_keypoint_detector, tactical_view_converter, ball_acquisition, pass_and_interception, drawers (visualization)
- **Data flow**: Video input → Object tracking → Team assignment → Ball possession analysis → Pass/interception detection → Court mapping → Tactical view output
- **Models**: Pre-trained YOLO models stored in `models/` directory (.pt files)
- **Stubs**: Cached data in `stubs/` for faster re-runs during development

## Code Style & Conventions
- **Import style**: Relative imports within modules (from .module import Class), absolute imports for external packages
- **Package structure**: Each major component as separate package with __init__.py exposing main classes
- **Naming**: snake_case for variables/functions, PascalCase for classes, descriptive method names
- **File organization**: Separate drawer classes for different visualizations, utility functions in utils/
- **Dependencies**: OpenCV for video processing, ultralytics for YOLO models, supervision for tracking utilities
- **Error handling**: Basic error handling, uses sys module for debugging
