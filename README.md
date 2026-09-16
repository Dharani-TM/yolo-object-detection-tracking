# YOLO Object Detection and Tracking

A real-time object detection and tracking project implemented using **Python, YOLO, and OpenCV**. The project detects objects through a camera/video source and provides smooth tracking using different OpenCV tracking algorithms.

I worked with the YOLO-based detection and tracking pipeline, configured the Python environment, installed the required libraries, executed the application, and tested real-time object detection using a webcam.

## Overview

This project combines **YOLO object detection** with **OpenCV object tracking** to provide real-time detection and tracking.
YOLO identifies objects in the video frames, while tracking algorithms help maintain smooth object movement between detection frames.
The application can detect multiple objects such as:

- Person
- Mobile phone
- Bottle
- Laptop
- Other objects supported by the YOLO model

## Features

- Real-time object detection
- Object tracking using OpenCV
- Webcam and video input support
- Multiple tracking algorithms
- Target selection and locking
- Bounding box visualization
- Object crop display
- Confidence threshold control
- Detection and tracking frequency control
- Real-time performance metrics
- Smooth bounding box movement

## Technologies Used

- **Python**
- **YOLO / Ultralytics**
- **OpenCV**
- **NumPy**
- **PyTorch**

## Project Structure

```text
yolo-object-detection-tracking/
│
├── python/
│   ├── objdet.py
│   ├── tracking_example.py
│   ├── yolov8n.pt
│   ├── yolo11n.pt
│   ├── yolo12n.pt
│   ├── yolo26n.pt
│   ├── yolo26s.pt
│   ├── dasiamrpn.onnx
│   ├── nanotrack_backbone.onnx
│   └── nanotrack_head.onnx
│
├── cpp/
│   ├── main.cpp
│   ├── Makefile
│   └── yolomodel_pt2onmx.py
│
├── .gitignore
├── LICENSE
├── README.md
└── README_CH.md


## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

See the [LICENSE](LICENSE) file for details.
