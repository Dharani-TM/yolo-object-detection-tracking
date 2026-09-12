# YOLO Object Detection & Tracking Demo

This project provides high-performance real-time object detection and tracking in both **Python** and **C++**. It utilizes [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for precise detection and integrates multiple OpenCV single-object trackers for robust target locking and smooth motion.

## Key Features (Python Version)

*   **Hybrid Detection & Tracking System**:
    *   **Low-Frequency Detection, High-Frequency Tracking**: Allows YOLO to run at a lower frequency (saving CPU/GPU resources) while using OpenCV feature trackers to fill the gaps at full camera FPS.
    *   **BG Track Toggle**: A dedicated switch to enable/disable background tracking. When **ON**, all detected objects are tracked smoothly. When **OFF**, background objects only update with YOLO detections, but the **Selected Target** remains smooth with high-frequency tracking.
*   **Visual Stability & Feedback**:
    *   **EMA Smoothing (Exponential Moving Average)**: Bounding boxes are filtered to eliminate jitter and "snapping" during detection corrections, resulting in butter-smooth motion.
    *   **Enhanced Target Locking**: Selected objects (via click or manual ROI) are highlighted with a **thick border (Thickness=4)** and prioritized for frame-by-frame tracking updates.
*   **Integrated Control Panel**:
    *   **Live Model Linkage**: Switching the tracker type (e.g., from NanoTrack to CSRT) instantly re-initializes all active trackers with the new algorithm.
    *   **Real-time Metrics**: Displays execution time (ms) and actual frequency (Hz) for both detection and tracking loops.
*   **Dual Window Monitoring**:
    *   **Main Window**: Displays the live video feed with smooth bounding boxes and UI controls.
    *   **Objects Window**: Displays a grid of real-time crops for every detected object, allowing for easy identification and selection.

## Folder Structure

```plaintext
├── python/
│   ├── objdet.py              # Main hybrid script (Detection + Tracking + UI)
│   ├── tracking_example.py    # Simplified OpenCV-only tracking example
│   ├── bytetrack.yaml         # YOLO internal tracker configuration
│   └── *.onnx / *.pt          # Model weights for trackers and YOLO
│
└── cpp/
    ├── main.cpp               # C++ detection demo (OpenCV DNN)
    ├── yolomodel_pt2onmx.py   # Model conversion tool (.pt -> .onnx)
    └── yolov8n.onnx           # ONNX model for C++ version
```

## Python Implementation Guide

### Requirements

*   Python 3.10+
*   `pip install ultralytics opencv-contrib-python numpy`

### Quick Start

1.  **Basic Run**:
    ```bash
    cd python
    python objdet.py
    ```
    *If no arguments are provided, an interactive menu will appear to help you select a camera mode.*

2.  **Hybrid Performance Mode (Recommended)**:
    ```bash
    # Run detection at 1Hz (low power) while tracking at 30Hz (high smoothness)
    python objdet.py --det-freq 1 --track-freq 30 --conf 0.5
    ```

### CLI Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--source` | Camera index or video file path | `0` or `video.mp4` |
| `--source_cfg` | Hardware resolution and FPS | `1280x720@30` |
| `--model` | YOLO model file path | `yolov8n.pt` |
| `--conf` | Confidence threshold (0~1) | `--conf 0.4` |
| `--det-freq` | Limit YOLO detection frequency (Hz) | `--det-freq 2.0` |
| `--track-freq` | Limit OpenCV tracking frequency (Hz) | `--track-freq 30.0` |

### On-Screen Controls

*   **Integrated/BG Track**: Toggle whether to track background objects or only the selected target.
*   **YOLO Mode**: Click on a bounding box in the feed or a crop in the `Objects` window to lock a target.
*   **Manual Mode**: Click and **drag the left mouse button** to draw a custom ROI. A yellow preview box appears during drawing.
*   **Engine Selection**: Click `NANO`, `DaSiam`, `CSRT`, `KCF`, or `MIL` to switch the tracking algorithm for all objects instantly.
*   **Keyboard**:
    *   `q`: Exit the application.

## Important Notes

*   **Smoothing Factor**: The responsiveness of the bounding boxes can be adjusted by changing `SMOOTH_ALPHA` in `objdet.py` (Default is 0.2).
*   **Wayland Support**: Automatically sets `QT_QPA_PLATFORM=xcb` for Linux environments using Wayland to ensure proper font rendering.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 
Please respect the individual licenses of third-party libraries and models.
