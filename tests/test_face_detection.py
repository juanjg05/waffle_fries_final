import cv2
import numpy as np
import argparse
import mediapipe as mp
import math
from enum import Enum
from typing import Tuple, Optional, Any
import time

class FaceDirection(Enum):
    TOWARDS = "towards"
    AWAY = "away"
    UNKNOWN = "unknown"

class FaceDetector:
    def __init__(self, face_angle_threshold: float = 30.0):
        """
        Initialize the face detector.
        
        Args:
            face_angle_threshold: Maximum angle (degrees) to consider face as towards
        """
        self.face_angle_threshold = face_angle_threshold
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # Changed to False for video
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For drawing face mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def _calculate_face_angle(self, landmarks) -> float:
        """
        Calculate the angle of the face relative to the camera.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            Angle in degrees (0 = facing camera, positive = turned right, negative = turned left)
        """
        try:
            # Use nose bridge and forehead points for better angle detection
            nose_bridge_top = landmarks[6]    # Top of nose bridge
            nose_bridge_bottom = landmarks[197]  # Bottom of nose bridge
            forehead = landmarks[151]  # Center of forehead
            
            # Calculate angle between nose bridge and vertical line from forehead
            dx = nose_bridge_bottom.x - nose_bridge_top.x
            dy = nose_bridge_bottom.y - nose_bridge_top.y
            
            # Calculate angle in degrees
            # When face is straight ahead, nose bridge should be vertical (small dx)
            angle = math.degrees(math.atan2(dx, dy)) * 2  # Multiply by 2 to amplify the angle
            
            return angle
            
        except Exception as e:
            print(f"Error calculating face angle: {str(e)}")
            return 0.0
    
    def detect_face_direction(self, image: np.ndarray) -> Tuple[FaceDirection, float, Optional[float], Optional[Any]]:
        """
        Detect if a face is facing towards or away from the robot.
        
        Args:
            image: RGB image containing the face
            
        Returns:
            Tuple of (FaceDirection, confidence, face_angle, landmarks)
        """
        try:
            # Use MediaPipe for face detection
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return FaceDirection.UNKNOWN, 0.0, None, None
            
            # Get landmarks for the first face
            landmarks = results.multi_face_landmarks[0]
            
            # Calculate face angle
            face_angle = self._calculate_face_angle(landmarks.landmark)
            
            # Determine face direction based on angle
            # Small angle (close to 0) means facing towards camera
            # Large angle means turned away
            if abs(face_angle) <= self.face_angle_threshold:
                return FaceDirection.TOWARDS, 0.9, face_angle, landmarks
            else:
                return FaceDirection.AWAY, 0.9, face_angle, landmarks
                
        except Exception as e:
            print(f"Error in face direction detection: {str(e)}")
            return FaceDirection.UNKNOWN, 0.0, None, None

def visualize_face_direction(image, face_direction, face_angle, confidence, landmarks=None):
    """Draw face direction information on the image."""
    # Create a copy of the image for drawing
    vis_image = image.copy()
    
    # Draw face mesh if landmarks are available
    if landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing.draw_landmarks(
            image=vis_image,
            landmark_list=landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
    
    # Add text showing face direction and angle
    text = f"Direction: {face_direction.value}"
    color = (0, 255, 0) if face_direction == FaceDirection.TOWARDS else (0, 0, 255)
    cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    text = f"Angle: {face_angle:.1f}°" if face_angle is not None else "Angle: Unknown"
    cv2.putText(vis_image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    text = f"Confidence: {confidence:.2f}"
    cv2.putText(vis_image, text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return vis_image

def process_video():
    """Process live video from webcam."""
    # Initialize the detector
    detector = FaceDetector()
    
    # Try different camera indices
    camera_indices = [0, 1, 2]  # Common webcam indices
    cap = None
    
    for idx in camera_indices:
        print(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Successfully opened camera {idx}")
            break
    
    if not cap or not cap.isOpened():
        print("Error: Could not open any webcam")
        print("Available camera devices:")
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera {idx} is available")
                cap.release()
        return
    
    print("\nPress 'q' to quit")
    print("Press 's' to save current frame")
    
    frame_count = 0
    fps = 0
    frame_times = []
    
    while True:
        # Start timing
        start_time = time.time()
        
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect face direction
        face_direction, confidence, face_angle, landmarks = detector.detect_face_direction(frame)
        
        # Visualize results
        vis_frame = visualize_face_direction(frame, face_direction, face_angle, confidence, landmarks)
        
        # Calculate and display FPS
        frame_times.append(time.time() - start_time)
        if len(frame_times) > 30:  # Keep last 30 frames
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))
        
        # Add FPS to display
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Face Direction Detection', vis_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            frame_count += 1
            filename = f"tests/data/capture_{frame_count}.jpg"
            cv2.imwrite(filename, vis_frame)
            print(f"Saved frame to {filename}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path: str):
    """Process a single image."""
    # Initialize the detector
    detector = FaceDetector()
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Detect face direction
    face_direction, confidence, face_angle, landmarks = detector.detect_face_direction(image)
    
    # Visualize results
    vis_image = visualize_face_direction(image, face_direction, face_angle, confidence, landmarks)
    
    # Save the result
    output_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(output_path, vis_image)
    print(f"Saved result to {output_path}")
    print(f"Face Direction: {face_direction.value}")
    print(f"Face Angle: {face_angle:.1f}°" if face_angle is not None else "Face Angle: Unknown")
    print(f"Confidence: {confidence:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Test face direction detection')
    parser.add_argument('--image', help='Path to the test image')
    parser.add_argument('--video', action='store_true', help='Use webcam for live video')
    args = parser.parse_args()
    
    if args.video:
        process_video()
    elif args.image:
        process_image(args.image)
    else:
        print("Please specify either --image or --video")
        parser.print_help()

if __name__ == "__main__":
    main() 