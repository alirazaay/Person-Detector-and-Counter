import cv2
import numpy as np
import torch
import sys
import queue
import multiprocessing
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFrame, QFileDialog, QStackedLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QMovie
import os
from collections import deque
from ultralytics import YOLO
import time

class PersonTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object IDs and their centroids
        self.disappeared = {}  # Dictionary to store number of frames object has been missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracked_ids = set()  # Set to store currently tracked IDs
        self.counted_ids = set()  # Set to store IDs that have been counted
        self.stable_frames = {}  # Dictionary to store how long each ID has been stable
        self.min_stable_frames = 10  # Minimum frames an object must be stable before counting
        self.counting_zone = None  # Zone where counting occurs
        self.last_frame_centroids = []  # Store last frame's centroids for motion compensation
        self.motion_threshold = 30  # Threshold for detecting significant camera motion
        self.counting_cooldown = {}  # Cooldown period for each counted ID
        
    def detect_camera_motion(self, current_centroids):
        if not self.last_frame_centroids:
            self.last_frame_centroids = current_centroids
            return False
            
        # Calculate average movement of all centroids
        total_movement = 0
        valid_pairs = 0
        
        for curr_cent in current_centroids:
            min_dist = float('inf')
            for last_cent in self.last_frame_centroids:
                dist = np.sqrt(np.sum((curr_cent - last_cent) ** 2))
                min_dist = min(min_dist, dist)
            if min_dist < self.max_distance:
                total_movement += min_dist
                valid_pairs += 1
                
        self.last_frame_centroids = current_centroids
        
        if valid_pairs == 0:
            return False
            
        avg_movement = total_movement / valid_pairs
        return avg_movement > self.motion_threshold
        
    def set_counting_zone(self, frame_shape):
        # Define counting zone (middle 60% of the frame)
        height, width = frame_shape[:2]
        zone_width = int(width * 0.6)
        zone_height = int(height * 0.6)
        x1 = (width - zone_width) // 2
        y1 = (height - zone_height) // 2
        self.counting_zone = (x1, y1, x1 + zone_width, y1 + zone_height)
        
    def is_in_counting_zone(self, centroid):
        if self.counting_zone is None:
            return True
        x, y = centroid
        x1, y1, x2, y2 = self.counting_zone
        return x1 <= x <= x2 and y1 <= y <= y2
        
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.stable_frames[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.tracked_ids:
            self.tracked_ids.remove(object_id)
        if object_id in self.stable_frames:
            del self.stable_frames[object_id]
        if object_id in self.counting_cooldown:
            del self.counting_cooldown[object_id]
            
    def update(self, rects, frame_shape):
        if self.counting_zone is None:
            self.set_counting_zone(frame_shape)
            
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.tracked_ids
            
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            
        # Check for camera motion
        if self.detect_camera_motion(input_centroids):
            # Reset stability counters during camera motion
            for object_id in self.stable_frames:
                self.stable_frames[object_id] = 0
            return self.objects, self.tracked_ids
            
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = np.sqrt(np.sum((object_centroids[i] - input_centroids[j]) ** 2))
                    
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update stability counter
                if self.is_in_counting_zone(input_centroids[col]):
                    self.stable_frames[object_id] += 1
                    if self.stable_frames[object_id] >= self.min_stable_frames:
                        self.tracked_ids.add(object_id)
                else:
                    self.stable_frames[object_id] = 0
                    if object_id in self.tracked_ids:
                        self.tracked_ids.remove(object_id)
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            for col in unused_cols:
                self.register(input_centroids[col])
                
        return self.objects, self.tracked_ids

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        
    def stop(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
        self.quit()
        
    def run(self):
        try:
            # Try different camera indices
            for camera_index in [0, 1, -1]:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if self.camera.isOpened():
                    print(f"Camera opened successfully with index {camera_index}")
                    break
            
            if not self.camera.isOpened():
                self.error_occurred.emit("Failed to open camera. Please check if camera is connected and not in use by another application.")
                return
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            # Clear any buffered frames
            for _ in range(5):
                self.camera.grab()
            
            # Verify camera settings
            width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized with resolution: {width}x{height} at {fps} FPS")
            
            if width == 0 or height == 0:
                self.error_occurred.emit("Failed to initialize camera with proper resolution.")
                return
            
            self.running = True
            frame_count = 0
            last_time = time.time()
            
            while self.running:
                # Read frame directly
                ret, frame = self.camera.read()
                if ret and frame is not None and frame.size > 0:
                    frame_count += 1
                    current_time = time.time()
                    
                    # Calculate and print FPS every second
                    if current_time - last_time >= 1.0:
                        print(f"Camera FPS: {frame_count}")
                        frame_count = 0
                        last_time = current_time
                    
                    # Resize frame here to reduce processing load
                    frame = cv2.resize(frame, (640, 480))
                    self.frame_ready.emit(frame)
                else:
                    print("Failed to read frame")
                    time.sleep(0.1)  # Add delay on error
                    continue
                    
        except Exception as e:
            print(f"Camera error: {str(e)}")
            self.error_occurred.emit(f"Camera error: {str(e)}")
        finally:
            self.stop()

class ProcessingThread(QThread):
    processed_frame = pyqtSignal(np.ndarray, int)
    processing_complete = pyqtSignal()
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.detection_threshold = 0.3
        self.tracker = PersonTracker(max_disappeared=30, max_distance=50)
        self.counted_ids = set()
        self.total_count = 0
        self.last_frame = None
        self.frame_skip = 0
        self.is_video = False
        self.video_capture = None
        self.use_half = torch.cuda.is_available()
        self.target_size = (640, 480)
        self.skip_frames = 2
        self.batch_size = 1
        
    def stop(self):
        self.running = False
        if self.video_capture is not None:
            self.video_capture.release()
        self.quit()
        
    def process_video(self, video_path):
        self.is_video = True
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            raise Exception("Failed to open video file")
            
        # Optimize video capture
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        while self.running and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % (self.skip_frames + 1) != 0:
                continue
                
            # Resize frame for faster processing
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Process frame
            self.process_single_frame(frame)
            
        self.video_capture.release()
        self.is_video = False
        self.processing_complete.emit()
        
    def process_single_frame(self, frame, is_image=False):
        try:
            if frame is not None and frame.size > 0:
                if frame.shape[1] > self.target_size[0] or frame.shape[0] > self.target_size[1]:
                    frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
                results = self.model(frame, 
                                  conf=self.detection_threshold, 
                                  classes=[0],
                                  verbose=False,
                                  half=self.use_half,
                                  imgsz=self.target_size,
                                  batch=self.batch_size)
                processed_frame = frame.copy()
                current_rects = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        current_rects.append((x1, y1, x2, y2))
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Person {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if is_image:
                    # For images, count all detected persons
                    count = len(current_rects)
                    self.total_count = count
                else:
                    # For video, use tracking logic
                    objects, tracked_ids = self.tracker.update(current_rects, frame.shape)
                    if self.tracker.counting_zone is not None:
                        x1, y1, x2, y2 = self.tracker.counting_zone
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(processed_frame, "Counting Zone", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    new_count = 0
                    current_time = time.time()
                    for object_id in tracked_ids:
                        if object_id not in self.counted_ids:
                            if object_id not in self.tracker.counting_cooldown or \
                               current_time - self.tracker.counting_cooldown[object_id] > 5.0:
                                new_count += 1
                                self.counted_ids.add(object_id)
                                self.tracker.counting_cooldown[object_id] = current_time
                    self.total_count += new_count
                    count = self.total_count
                cv2.putText(processed_frame, f"Count: {count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.processed_frame.emit(processed_frame, count)
        except Exception as e:
            print(f"Processing error: {str(e)}")
            
    def run(self):
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.process_single_frame(frame)
            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue

class PersonDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize model with optimized settings
        self.model = YOLO('yolov8n.pt')
        if torch.cuda.is_available():
            self.model.to('cuda')
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
            print("Using GPU for inference")
        else:
            print("Using CPU for inference")
        
        # Initialize variables
        self.person_count = 0
        self.total_count = 0
        
        # Initialize processing thread
        self.processing_thread = ProcessingThread(self.model)
        
        # Connect signals
        self.processing_thread.processed_frame.connect(self.update_display)
        self.processing_thread.processing_complete.connect(self.handle_processing_complete)
        
        # Start processing thread
        self.processing_thread.start()
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Person Detection and Counter")
        self.setMinimumSize(1280, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create header with gradient background
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #2C3E50, stop:1 #3498DB);
                border-radius: 10px;
                padding: 5px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        
        header = QLabel("Person Detection and Counter")
        header.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: white;
                padding: 5px;
            }
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(header)
        main_layout.addWidget(header_frame)
        
        # --- Control panel with only upload image and camera icon ---
        control_panel = QFrame()
        control_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(44, 62, 80, 0.9);
                border-radius: 10px;
                padding: 5px;
            }
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(10)

        # Camera icon + Upload image button
        camera_button_container = QWidget()
        camera_button_layout = QHBoxLayout(camera_button_container)
        camera_button_layout.setSpacing(5)
        # Camera icon
        camera_icon = QLabel()
        camera_pixmap = QPixmap("camera.png")
        if camera_pixmap.isNull():
            camera_pixmap = QPixmap(32, 32)
            camera_pixmap.fill(Qt.GlobalColor.transparent)
        camera_icon.setPixmap(camera_pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio))
        # Upload image button
        self.upload_image_button = QPushButton("Upload Image")
        self.upload_image_button.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                border: none;
                padding: 8px 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27AE60;
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: #219A52;
            }
        """)
        self.upload_image_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_image_button.clicked.connect(self.upload_image)
        camera_button_layout.addWidget(camera_icon)
        camera_button_layout.addWidget(self.upload_image_button)
        control_layout.addWidget(camera_button_container)
        control_layout.addStretch()
        # Total count display (unchanged)
        count_frame = QFrame()
        count_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(52, 152, 219, 0.1);
                border: 2px solid #3498DB;
                border-radius: 8px;
                padding: 5px;
            }
        """)
        count_layout = QHBoxLayout(count_frame)
        count_layout.setSpacing(10)
        total_count = QFrame()
        total_count.setStyleSheet("""
            QFrame {
                background-color: rgba(46, 204, 113, 0.1);
                border: 2px solid #2ECC71;
                border-radius: 8px;
                padding: 5px;
                min-width: 150px;
            }
        """)
        total_layout = QVBoxLayout(total_count)
        total_label = QLabel("Total Count")
        total_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #27AE60;
                font-weight: bold;
            }
        """)
        self.total_count_label = QLabel("0")
        self.total_count_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #27AE60;
            }
        """)
        total_layout.addWidget(total_label, alignment=Qt.AlignmentFlag.AlignCenter)
        total_layout.addWidget(self.total_count_label, alignment=Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(total_count)
        main_layout.addWidget(control_panel)
        
        # --- NEW: Titles row for both boxes ---
        titles_row = QHBoxLayout()
        left_label_title = QLabel("Uploaded File (Raw)")
        left_label_title.setStyleSheet("font-size: 16px; color: #ecf0f1; font-weight: bold;")
        left_label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_label_title = QLabel("Detection Output")
        right_label_title.setStyleSheet("font-size: 16px; color: #ecf0f1; font-weight: bold;")
        right_label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titles_row.addWidget(left_label_title)
        titles_row.addWidget(right_label_title)
        main_layout.addLayout(titles_row)
        
        # --- Split display area into two boxes ---
        display_split = QHBoxLayout()
        display_split.setSpacing(10)
        # Left: Raw input
        left_frame = QFrame()
        left_frame.setStyleSheet("QFrame { background-color: #34495E; border-radius: 10px; padding: 5px; }")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(5, 5, 5, 5)
        self.input_label = QLabel()
        self.input_label.setStyleSheet("background-color: #2C3E50; border-radius: 8px; padding: 5px;")
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_label.setMinimumSize(480, 360)
        left_layout.addWidget(self.input_label)
        display_split.addWidget(left_frame, stretch=1)
        # Right: Detection output (overlay spinner/status)
        right_frame = QFrame()
        right_frame.setStyleSheet("QFrame { background-color: #34495E; border-radius: 10px; padding: 5px; }")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(5, 5, 5, 5)
        # Overlay layout for detection output
        overlay_widget = QWidget()
        overlay_layout = QStackedLayout(overlay_widget)
        # Detection output image label
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: #2C3E50; border-radius: 8px; padding: 5px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        overlay_layout.addWidget(self.video_label)
        # Spinner/status overlay (top center)
        spinner_status_widget = QWidget(self.video_label)
        spinner_status_layout = QHBoxLayout(spinner_status_widget)
        spinner_status_layout.setContentsMargins(0, 10, 0, 0)
        spinner_status_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.processing_anim = QLabel()
        self.processing_anim.setFixedSize(32, 32)
        self.processing_movie = QMovie("spinner.gif")
        self.processing_anim.setMovie(self.processing_movie)
        self.processing_anim.hide()
        self.processing_status = QLabel("")
        self.processing_status.setStyleSheet("font-size: 16px; color: #f1c40f; font-weight: bold;")
        spinner_status_layout.addWidget(self.processing_anim)
        spinner_status_layout.addWidget(self.processing_status)
        overlay_layout.addWidget(spinner_status_widget)
        overlay_layout.setCurrentIndex(0)  # Show image by default
        right_layout.addWidget(overlay_widget)
        display_split.addWidget(right_frame, stretch=1)
        main_layout.addLayout(display_split, stretch=1)
        
        # Create status bar with modern style
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2C3E50;
                color: white;
                padding: 5px;
                font-size: 12px;
            }
        """)
        self.statusBar().showMessage("Ready")
        
    def show_spinner(self, show=True):
        # Helper to show/hide spinner overlay
        overlay_widget = self.video_label.parentWidget()
        overlay_layout = overlay_widget.layout()
        if show:
            overlay_layout.setCurrentIndex(1)
            self.processing_anim.show()
            self.processing_movie.start()
        else:
            overlay_layout.setCurrentIndex(0)
            self.processing_anim.hide()
            self.processing_movie.stop()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        if file_path:
            try:
                self.total_count = 0
                self.total_count_label.setText("0")
                self.processing_thread.counted_ids.clear()
                self.processing_thread.total_count = 0
                self.processing_thread.tracker.counted_ids.clear()
                self.processing_thread.tracker.counting_cooldown.clear()
                image = cv2.imread(file_path)
                if image is not None:
                    # Show raw image immediately in left box, scaled to fit
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    scaled_pixmap = pixmap.scaled(self.input_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.input_label.setPixmap(scaled_pixmap)
                    self.input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    # --- Show spinner and status immediately ---
                    self.processing_status.setText("Processing image...")
                    self.show_spinner(True)
                    QApplication.processEvents()  # Force UI update
                    # Store the original image for resizing on window resize
                    self._last_raw_image = pixmap
                    # Pass is_image=True to use direct counting
                    self.processing_thread.process_single_frame(image, is_image=True)
                    self.statusBar().showMessage(f"Image processed: {file_path}")
                else:
                    self.statusBar().showMessage("Failed to load image")
            except Exception as e:
                self.statusBar().showMessage(f"Error processing image: {str(e)}")
                
    def handle_processing_complete(self):
        self.statusBar().showMessage("Processing complete")
        
    def update_display(self, frame, count):
        try:
            # Convert frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            q_image = q_image.rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Store the last detection output for resizing on window resize
            self._last_detection_pixmap = pixmap
            self.total_count = count
            self.total_count_label.setText(str(self.total_count))
            # --- Hide spinner and status after processing is done ---
            self.processing_status.setText("")
            self.show_spinner(False)
        except Exception as e:
            print(f"Display update error: {str(e)}")
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale the raw image in the left box
        if hasattr(self, '_last_raw_image') and self.input_label:
            scaled_pixmap = self._last_raw_image.scaled(self.input_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.input_label.setPixmap(scaled_pixmap)
        # Rescale the detection output in the right box
        if hasattr(self, '_last_detection_pixmap') and self.video_label:
            scaled_pixmap = self._last_detection_pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
                
    def closeEvent(self, event):
        self.processing_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for modern look
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1A1A1A;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QMenuBar {
            background-color: #2C3E50;
            color: white;
        }
        QMenuBar::item:selected {
            background-color: #3498DB;
        }
        QMenu {
            background-color: #2C3E50;
            color: white;
        }
        QMenu::item:selected {
            background-color: #3498DB;
        }
        QScrollBar:vertical {
            border: none;
            background-color: #2C3E50;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #3498DB;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            border: none;
            background-color: #2C3E50;
            height: 10px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #3498DB;
            min-width: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)
    
    window = PersonDetector()
    window.show()
    sys.exit(app.exec()) 