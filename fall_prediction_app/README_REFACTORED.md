# Refactored Fall Detection Application

This is a refactored version of the original `webcam_feed.py` that follows SOLID principles and provides better code organization.

## Architecture

The application is now split into focused service classes, each with a single responsibility:

### Services

1. **CameraService** (`services/camera_service.py`)
   - Handles camera initialization and frame capture
   - Supports single and multi-camera setups
   - Manages camera resources

2. **ObjectDetector** (`services/object_detector.py`)
   - Handles YOLOv8 object detection
   - Separates person detections from other objects
   - Draws bounding boxes for detected objects

3. **PoseAnalyzer** (`services/pose_analyzer.py`)
   - Handles MediaPipe pose estimation
   - Calculates fall detection metrics (trunk angle, NSAR, plumb angles)
   - Determines fall status based on multiple criteria

4. **AlertService** (`services/alert_service.py`)
   - Manages external alerts (Telegram, Azure Foundry)
   - Handles alert timing and message formatting
   - Provides clean separation of alert logic

5. **DisplayService** (`services/display_service.py`)
   - Handles all UI rendering and HUD display
   - Manages FPS calculation and metrics buffering
   - Provides fullscreen display functionality

6. **StatisticsService** (`services/statistics_service.py`)
   - Tracks performance metrics and statistics
   - Calculates latency and detection rates
   - Provides clean statistics interface

### Main Application

- **FallDetectionApp** (`refactored_webcam_feed.py`)
  - Orchestrates all services
  - Manages the main processing loop
  - Handles application lifecycle

## Benefits of Refactoring

1. **Single Responsibility Principle**: Each class has one clear purpose
2. **Open/Closed Principle**: Easy to extend without modifying existing code
3. **Dependency Inversion**: Services depend on abstractions, not concrete implementations
4. **Better Testability**: Each service can be tested independently
5. **Improved Maintainability**: Changes to one aspect don't affect others
6. **Cleaner Code**: Much easier to read and understand

## Usage

### Basic Usage
```bash
cd fall_prediction_app
python run_refactored.py
```

### With Command Line Arguments
```bash
python run_refactored.py --sources "0,1" --multi-view --pose-complexity 2
```

### Arguments
- `--sources`: Comma-separated list of video sources (default: "0")
- `--multi-view`: Enable multi-camera processing
- `--pose-complexity`: MediaPipe pose complexity (0=lite, 1=full, 2=heavy)

## Environment Variables

Make sure to set these environment variables for alerts:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID

## File Structure

```
fall_prediction_app/
├── services/
│   ├── __init__.py
│   ├── camera_service.py
│   ├── object_detector.py
│   ├── pose_analyzer.py
│   ├── alert_service.py
│   ├── display_service.py
│   └── statistics_service.py
├── refactored_webcam_feed.py
├── run_refactored.py
├── requirements.txt
└── README_REFACTORED.md
```

## Migration from Original

The refactored version maintains the same functionality as the original but with much better code organization. All the original features are preserved:

- Real-time person detection with YOLOv8
- Pose estimation with MediaPipe
- Multiple fall detection algorithms (trunk angle, NSAR, plumb angles)
- External alerts (Telegram, Azure Foundry)
- Fullscreen display with HUD
- Multi-camera support
- Performance statistics

## Future Enhancements

The modular design makes it easy to add new features:

- Add new fall detection algorithms by extending PoseAnalyzer
- Add new alert services by extending AlertService
- Add new display modes by extending DisplayService
- Add new camera types by extending CameraService 