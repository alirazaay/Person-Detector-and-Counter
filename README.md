# Person Detection and Counter

This project implements real-time person detection and counting using YOLOv9. It can detect and count people both from a webcam feed and from uploaded images.

## Features

- Real-time person detection using webcam
- Person detection in uploaded images
- Person counting with visual bounding boxes
- Simple and intuitive GUI interface

## Requirements

- Python 3.8 or higher
- Webcam (for real-time detection)
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. The application will automatically download the YOLOv9 model on first run

## Usage

1. Run the application:
   ```bash
   python person_detector.py
   ```

2. Using the application:
   - Click "Start Camera" to begin real-time person detection using your webcam
   - Click "Upload Image" to select and process an image file
   - The person count will be displayed below the video/image
   - Green bounding boxes will be drawn around detected people

## Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- GIF

## Notes

- The application uses YOLOv9 for person detection
- Real-time detection performance may vary depending on your hardware
- Make sure you have good lighting conditions for better detection results 