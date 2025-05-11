# Person Detection and Counter

---

## Title Slide

**Person Detection and Counter**  
*Ali Raza*  
*June 2024*

---

## Introduction

- Detects and counts people in images using AI (YOLOv8)
- Modern, user-friendly desktop application
- Fast, accurate, and visually appealing
- Built for real-time and batch image analysis

---

## Key Features

- Upload and process images instantly
- Accurate person detection and counting
- Modern, responsive UI (PyQt6)
- Download/save detection results
- Copy count to clipboard
- Drag-and-drop image upload
- Recent files gallery
- Detection summary panel
- Dark/light theme toggle
- Zoom and pan output
- Show/hide detection boxes
- Instant image enhancement
- Hotkey support for all major actions

---

## System Architecture

- **Frontend:** PyQt6 GUI
- **Backend:** YOLOv8 model (Ultralytics)
- **Processing:** Separate thread for detection
- **Display:** Dual-pane (raw & detection output)
- **Optimizations:** Frame resizing, efficient memory use, GPU/CPU support

---

## Detection Pipeline

1. **Image Upload** (button or drag-and-drop)
2. **Preprocessing** (resize, enhance if selected)
3. **YOLOv8 Inference** (detect people)
4. **Postprocessing** (draw boxes, count, summary)
5. **Display Results** (side-by-side, instant feedback)
6. **Save/Copy/Share** (output, count, gallery)

---

## User Interface

![Main UI](screenshot1.png)

- Clean, modern layout
- Upload image button with camera icon
- Total count display
- Dual-pane for raw and detection output

---

## Detection Example

![Detection Example](screenshot2.png)

- Left: Uploaded image (raw)
- Right: Detection output with bounding boxes and count
- Fast, accurate results

---

## Performance & Advanced Features

- Optimized for speed (frame skipping, resizing, efficient threading)
- Drag-and-drop, gallery, and summary panel for productivity
- Download, copy, and hotkey support for quick actions
- Theme toggle for accessibility
- Zoom/pan and show/hide boxes for better inspection
- Instant image enhancement for low-light images

---

## Possible Extensions

- Multi-class detection (cars, bags, etc.)
- Face detection and blurring
- Gender/age estimation
- Pose estimation
- Mask/helmet/uniform detection
- Crowd density heatmap
- Region of interest (ROI) selection
- Counting by zone
- Batch image processing
- Export detection data (CSV/JSON)

---

## Conclusion & Contact

- **Person Detection and Counter** is a fast, modern, and extensible tool for AI-powered people counting.
- Built by **Ali Raza**
- [GitHub Repository/Contact Info Placeholder]

--- 