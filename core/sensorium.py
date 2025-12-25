"""
SpatialFlow Sensorium Module
=============================

Encapsulates MediaPipe hand tracking with optimized, non-blocking webcam capture.
Applies One Euro Filter to all landmarks for God-Level tracking stability.

Updated for MediaPipe 0.10.x Tasks API.

Key Features:
    - Threaded camera capture for 60 FPS main loop
    - 21-landmark filtering per coordinate (63 filters total per hand)
    - Gesture detection (Pinch, Fist, Point)
    - Coordinate mapping from normalized to Ursina world space
"""

import cv2
import numpy as np
import threading
import time
import os
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from math import tan, radians

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .signals import OneEuroFilter, OneEuroFilter3D, SchmittTrigger, GestureDetector


# MediaPipe landmark indices for reference
class HandLandmark(Enum):
    """MediaPipe hand landmark indices"""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class HandData:
    """
    Processed hand data with filtered landmarks and gesture state.
    """
    # Filtered landmark positions (21 landmarks, each with x, y, z)
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    
    # Raw MediaPipe landmarks (for comparison/debugging)
    landmarks_raw: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    
    # Gesture states
    is_pinching: bool = False
    is_fist: bool = False
    is_pointing: bool = False
    is_peace: bool = False

    
    # Derived positions
    index_tip: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    thumb_tip: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    palm_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Confidence and handedness
    handedness: str = "Right"
    score: float = 0.0
    
    # Detection status
    detected: bool = False
    timestamp: float = 0.0
    
    def get_landmark(self, idx: int) -> Tuple[float, float, float]:
        """Get a specific landmark by index"""
        return tuple(self.landmarks[idx])
    
    def get_landmark_by_name(self, name: HandLandmark) -> Tuple[float, float, float]:
        """Get a specific landmark by enum name"""
        return tuple(self.landmarks[name.value])


class CoordinateMapper:
    """
    Maps normalized MediaPipe coordinates to Ursina world space.
    
    MediaPipe outputs coordinates in [0, 1] normalized space.
    This class transforms them to Ursina's 3D world space using
    the camera's FOV and aspect ratio.
    """
    
    def __init__(
        self,
        camera_fov: float = 60.0,
        aspect_ratio: float = 16/9,
        z_plane: float = -10.0
    ):
        self._fov = camera_fov
        self._aspect = aspect_ratio
        self._z_plane = z_plane
        self._update_projection()
    
    def _update_projection(self):
        """Update projection parameters based on camera settings"""
        self._half_height = abs(self._z_plane) * tan(radians(self._fov / 2))
        self._half_width = self._half_height * self._aspect
    
    def update_camera(self, fov: float, aspect_ratio: float, z_plane: float):
        """Update camera parameters"""
        self._fov = fov
        self._aspect = aspect_ratio
        self._z_plane = z_plane
        self._update_projection()
    
    def normalized_to_world(
        self,
        norm_x: float,
        norm_y: float,
        z_depth: Optional[float] = None,
        mirror_x: bool = True
    ) -> Tuple[float, float, float]:
        """
        Transform normalized coordinates to Ursina world space.
        """
        z = z_depth if z_depth is not None else self._z_plane
        
        # Transform from [0, 1] to [-1, 1]
        x_centered = (norm_x - 0.5) * 2.0
        y_centered = (0.5 - norm_y) * 2.0  # Flip Y (webcam Y is inverted)
        
        # Mirror X for selfie view
        if mirror_x:
            x_centered = -x_centered
        
        # Scale to world coordinates
        world_x = x_centered * self._half_width
        world_y = y_centered * self._half_height
        
        return (world_x, world_y, z)


class HandTracker:
    """
    Main hand tracking class with MediaPipe Tasks API and One Euro filtering.
    
    Features:
        - Non-blocking webcam capture
        - 63 One Euro Filters (21 landmarks × 3 coordinates)
        - Gesture detection with hysteresis
        - Coordinate mapping to Ursina world space
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
        filter_min_cutoff: float = 0.5,  # Lowered from 1.0 for smoother slow movement
        filter_beta: float = 0.05,       # Lowered from 0.2 to significantly reduce jitter
        camera_fov: float = 60.0
    ):
        """
        Initialize the hand tracker.
        """
        self._camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Find model path
        if model_path is None:
            # Look for model in current directory or script directory
            possible_paths = [
                'hand_landmarker.task',
                os.path.join(os.path.dirname(__file__), '..', 'hand_landmarker.task'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hand_landmarker.task')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found. Please download it from:\n"
                f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
                f"and place it in the project directory."
            )
        
        self._model_path = model_path
        print(f"[HandTracker] Using model: {model_path}")
        
        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Filters for each landmark (21 landmarks × 3D filter each)
        self._landmark_filters: List[OneEuroFilter3D] = [
            OneEuroFilter3D(filter_min_cutoff, filter_beta)
            for _ in range(21)
        ]
        
        # Gesture detector
        self._gesture_detector = GestureDetector(
            pinch_low=0.06,   # Very easy to trigger (instant)
            pinch_high=0.07,  # Very quick release (minimal hysteresis)
            fist_threshold=0.15
        )



        
        # Coordinate mapper
        self._coord_mapper = CoordinateMapper(
            camera_fov=camera_fov,
            aspect_ratio=16/9,
            z_plane=-10.0
        )
        
        # State
        self._hand_data = HandData()
        self._frame_count = 0
        
        # Debug
        self._show_debug = False
        self._debug_frame: Optional[np.ndarray] = None
    
    @property
    def hand_data(self) -> HandData:
        return self._hand_data
    
    @property
    def is_hand_detected(self) -> bool:
        return self._hand_data.detected
    
    @property
    def coordinate_mapper(self) -> CoordinateMapper:
        return self._coord_mapper
    
    @property
    def debug_frame(self) -> Optional[np.ndarray]:
        return self._debug_frame
    
    def start(self) -> bool:
        """Start the hand tracker."""
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            print(f"[HandTracker] Failed to open camera {self._camera_index}")
            return False
        
        # Get camera properties
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Update coordinate mapper with actual aspect ratio
        aspect_ratio = width / height if height > 0 else 16/9
        self._coord_mapper.update_camera(60.0, aspect_ratio, -10.0)
        
        print(f"[HandTracker] Camera opened: {width}x{height}")
        return True
    
    def stop(self):
        """Stop the hand tracker."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._show_debug:
            cv2.destroyAllWindows()
        print("[HandTracker] Stopped")
    
    def set_debug_visualization(self, enabled: bool):
        """Enable or disable debug visualization window."""
        self._show_debug = enabled
        if not enabled:
            cv2.destroyAllWindows()
    
    def update(self) -> HandData:
        """
        Process the latest camera frame and update hand data.
        """
        if self._cap is None or not self._cap.isOpened():
            self._hand_data.detected = False
            return self._hand_data
        
        # Read frame
        ret, frame = self._cap.read()
        if not ret:
            self._hand_data.detected = False
            return self._hand_data
            
        # Store for external use (AR background)
        # Flip for selfie view if needed (User will see themselves mirrored)
        self.latest_frame = cv2.flip(frame, 1) # Mirror for intuitive interaction
        frame = self.latest_frame # Use mirrored frame for processing too

        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands
        results = self._hand_landmarker.detect(mp_image)
        
        current_time = time.time()
        
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            landmarks = results.hand_landmarks[0]  # First hand
            
            # Get handedness
            if results.handedness and len(results.handedness) > 0:
                handedness = results.handedness[0][0]
                self._hand_data.handedness = handedness.category_name
                self._hand_data.score = handedness.score
            
            # Process landmarks
            raw_landmarks = np.zeros((21, 3))
            filtered_landmarks = np.zeros((21, 3))
            
            for i, lm in enumerate(landmarks):
                # Store raw
                raw_landmarks[i] = [lm.x, lm.y, lm.z]
                
                # Apply One Euro Filter
                fx, fy, fz = self._landmark_filters[i].filter(lm.x, lm.y, lm.z)
                filtered_landmarks[i] = [fx, fy, fz]
            
            self._hand_data.landmarks = filtered_landmarks
            self._hand_data.landmarks_raw = raw_landmarks
            
            # Extract key positions
            idx_tip = HandLandmark.INDEX_TIP.value
            thumb_tip = HandLandmark.THUMB_TIP.value
            
            self._hand_data.index_tip = tuple(filtered_landmarks[idx_tip])
            self._hand_data.thumb_tip = tuple(filtered_landmarks[thumb_tip])
            
            # Calculate palm center
            palm_indices = [0, 5, 9, 13, 17]
            palm_center = np.mean(filtered_landmarks[palm_indices], axis=0)
            self._hand_data.palm_center = tuple(palm_center)
            
            # Calculate pinch distance
            pinch_dist = np.linalg.norm(
                filtered_landmarks[idx_tip] - filtered_landmarks[thumb_tip]
            )
            
            # Update gesture detection
            self._hand_data.is_pinching = self._gesture_detector.update_pinch(pinch_dist)
            
            # Detect fist
            fingertip_indices = [4, 8, 12, 16, 20]
            fingertip_positions = filtered_landmarks[fingertip_indices]
            avg_dist = np.mean(np.linalg.norm(fingertip_positions - palm_center, axis=1))
            self._hand_data.is_fist = avg_dist < 0.12
            
            # Detect pointing
            index_tip_dist = np.linalg.norm(filtered_landmarks[8] - palm_center)
            middle_tip_dist = np.linalg.norm(filtered_landmarks[12] - palm_center)
            other_tips_dist = np.mean([
                np.linalg.norm(filtered_landmarks[i] - palm_center)
                for i in [4, 16, 20]  # Thumb, Ring, Pinky
            ])
            
            self._hand_data.is_pointing = (
                index_tip_dist > 0.15 and middle_tip_dist < 0.12 and other_tips_dist < 0.12
            )
            
            # Detect peace sign
            others_peace_dist = np.mean([
                np.linalg.norm(filtered_landmarks[i] - palm_center)
                for i in [4, 16, 20]
            ])
            self._hand_data.is_peace = self._gesture_detector.update_peace(
                index_tip_dist, middle_tip_dist, others_peace_dist
            )

            
            self._hand_data.detected = True
            self._hand_data.timestamp = current_time
            
            # Debug visualization
            if self._show_debug:
                self._create_debug_frame(frame, filtered_landmarks)
        else:
            self._hand_data.detected = False
        
        # Show debug window
        if self._show_debug and self._debug_frame is not None:
            cv2.imshow('SpatialFlow - Hand Tracking Debug', self._debug_frame)
            cv2.waitKey(1)
        
        self._frame_count += 1
        return self._hand_data
    
    def get_cursor_world_position(self, z_depth: float = -10.0) -> Tuple[float, float, float]:
        """Get the cursor position in world space."""
        if not self._hand_data.detected:
            return (0.0, 0.0, z_depth)
        
        idx_x, idx_y, _ = self._hand_data.index_tip
        return self._coord_mapper.normalized_to_world(idx_x, idx_y, z_depth)
    
    def get_palm_world_position(self, z_depth: float = -10.0) -> Tuple[float, float, float]:
        """Get the palm center position in world space."""
        if not self._hand_data.detected:
            return (0.0, 0.0, z_depth)
        
        palm_x, palm_y, _ = self._hand_data.palm_center
        return self._coord_mapper.normalized_to_world(palm_x, palm_y, z_depth)
    
    def _create_debug_frame(self, frame: np.ndarray, landmarks: np.ndarray):
        """Create debug visualization frame."""
        debug = frame.copy()
        h, w = debug.shape[:2]
        
        # Draw landmarks
        for i, lm in enumerate(landmarks):
            x = int(lm[0] * w)
            y = int(lm[1] * h)
            cv2.circle(debug, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for start, end in connections:
            x1 = int(landmarks[start][0] * w)
            y1 = int(landmarks[start][1] * h)
            x2 = int(landmarks[end][0] * w)
            y2 = int(landmarks[end][1] * h)
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add gesture indicators
        pinch_color = (0, 255, 0) if self._hand_data.is_pinching else (0, 100, 100)
        cv2.putText(debug, f"PINCH: {'YES' if self._hand_data.is_pinching else 'NO'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pinch_color, 2)
        
        fist_color = (0, 255, 0) if self._hand_data.is_fist else (0, 100, 100)
        cv2.putText(debug, f"FIST: {'YES' if self._hand_data.is_fist else 'NO'}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fist_color, 2)
        
        peace_color = (0, 255, 0) if self._hand_data.is_peace else (0, 100, 100)
        cv2.putText(debug, f"PEACE: {'YES' if self._hand_data.is_peace else 'NO'}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, peace_color, 2)
        
        cv2.putText(debug, f"Hand: {self._hand_data.handedness}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        self._debug_frame = debug
    
    def screen_to_world(
        self,
        norm_x: float,
        norm_y: float,
        z_depth: float = -10.0
    ) -> Tuple[float, float, float]:
        """Transform MediaPipe normalized coordinates to Ursina world space."""
        return self._coord_mapper.normalized_to_world(norm_x, norm_y, z_depth)


# Testing
if __name__ == "__main__":
    print("Testing HandTracker with new MediaPipe API...")
    
    tracker = HandTracker(camera_index=0)
    tracker.set_debug_visualization(True)
    
    if tracker.start():
        print("Hand tracker started. Press 'q' to quit.")
        
        try:
            while True:
                hand_data = tracker.update()
                
                if hand_data.detected:
                    cursor_pos = tracker.get_cursor_world_position()
                    print(f"Cursor: ({cursor_pos[0]:.2f}, {cursor_pos[1]:.2f}) | "
                          f"Pinch: {hand_data.is_pinching} | "
                          f"Fist: {hand_data.is_fist}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            tracker.stop()
    else:
        print("Failed to start hand tracker")
