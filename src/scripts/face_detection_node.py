#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Any, Dict
import json
import os
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        """Initialize the face detector with MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_face_direction(self, frame: np.ndarray) -> Dict:
        """
        Detect face direction using MediaPipe Face Mesh.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing face direction information
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'direction': 'unknown',
                'angle': None,
                'confidence': 0.0,
                'is_spoken_to': False
            }
            
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get nose bridge landmarks (indices 1 and 4)
        nose_bridge = [
            face_landmarks.landmark[1],  # Top of nose bridge
            face_landmarks.landmark[4]   # Bottom of nose bridge
        ]
        
        # Calculate angle
        dx = nose_bridge[1].x - nose_bridge[0].x
        dy = nose_bridge[1].y - nose_bridge[0].y
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Determine direction
        if abs(angle) < 30:  # Threshold for "towards"
            direction = 'towards'
            is_spoken_to = True
        else:
            direction = 'away'
            is_spoken_to = False
            
        return {
            'direction': direction,
            'angle': float(angle),
            'confidence': 0.9,
            'is_spoken_to': is_spoken_to
        }

class FaceDetectionProcessor:
    def __init__(self, output_dir="face_detection_results"):
        """
        Initialize face detection processor.
        
        Args:
            output_dir: Directory to save face detection results
        """
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Initialize video capture
        self.cap = None
        self.video_width = 640
        self.video_height = 480
        self.fps = 30
        
        logger.info('Face detection processor initialized')
        
    def process_video_file(self, video_file: str) -> Dict:
        """
        Process a video file for face detection.
        
        Args:
            video_file: Path to video file to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_file}")
                return {'error': 'Failed to open video file'}
            
            # Get video properties
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Initialize results
            face_results = []
            frame_count = 0
            
            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get timestamp
                timestamp = frame_count / self.fps
                
                # Detect face direction
                face_info = self.face_detector.detect_face_direction(frame)
                
                # Add timestamp to result
                result = {
                    'frame': frame_count,
                    'timestamp': timestamp,
                    **face_info
                }
                
                face_results.append(result)
                frame_count += 1
            
            # Close video file
            cap.release()
            
            # Save results to file
            result_filename = os.path.join(
                self.output_dir, 
                os.path.splitext(os.path.basename(video_file))[0] + "_face_detection.json"
            )
            
            with open(result_filename, 'w') as f:
                json.dump({
                    'filename': os.path.basename(video_file),
                    'processed_at': datetime.now().isoformat(),
                    'frame_count': frame_count,
                    'fps': self.fps,
                    'resolution': f"{self.video_width}x{self.video_height}",
                    'face_results': face_results
                }, f, indent=2)
            
            logger.info(f"Results saved to {result_filename}")
            
            # Return summary
            return {
                'filename': os.path.basename(video_file),
                'frame_count': frame_count,
                'fps': self.fps,
                'resolution': f"{self.video_width}x{self.video_height}",
                'face_results_count': len(face_results)
            }
            
        except Exception as e:
            logger.error(f'Error processing video: {str(e)}')
            return {'error': str(e)}
    
    def start_webcam_detection(self, show_preview=True):
        """
        Start face detection from webcam.
        
        Args:
            show_preview: Whether to show the webcam preview window
        """
        try:
            # Open webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Error opening webcam")
                return False
            
            # Get webcam properties
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Webcam opened: {self.video_width}x{self.video_height} @ {self.fps}fps")
            
            # Initialize results
            face_results = []
            start_time = time.time()
            
            # Process frames
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Get timestamp
                timestamp = time.time() - start_time
                
                # Detect face direction
                face_info = self.face_detector.detect_face_direction(frame)
                
                # Add timestamp to result
                result = {
                    'timestamp': timestamp,
                    **face_info
                }
                
                face_results.append(result)
                
                # Display the face direction on the frame
                if show_preview:
                    direction = face_info['direction']
                    angle = face_info['angle']
                    confidence = face_info['confidence']
                    
                    color = (0, 255, 0) if direction == 'towards' else (0, 0, 255)
                    text = f"Direction: {direction}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    if angle is not None:
                        text = f"Angle: {angle:.1f}°"
                        cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    cv2.imshow('Face Detection', frame)
                    
                    # Exit on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Close webcam and windows
            if self.cap:
                self.cap.release()
            
            if show_preview:
                cv2.destroyAllWindows()
            
            # Save results to file
            result_filename = os.path.join(
                self.output_dir, 
                f"webcam_face_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(result_filename, 'w') as f:
                json.dump({
                    'start_time': datetime.fromtimestamp(start_time).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration': time.time() - start_time,
                    'resolution': f"{self.video_width}x{self.video_height}",
                    'face_results': face_results[-100:]  # Save only the last 100 results
                }, f, indent=2)
            
            logger.info(f"Results saved to {result_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f'Error in webcam detection: {str(e)}')
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            return False

def main():
    """Run face detection processor in standalone mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face detection processor")
    parser.add_argument("--video-file", type=str, help="Path to video file to process")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for face detection")
    parser.add_argument("--output-dir", type=str, default="face_detection_results", 
                        help="Directory to save face detection results")
    
    args = parser.parse_args()
    
    processor = FaceDetectionProcessor(output_dir=args.output_dir)
    
    if args.webcam:
        print("Starting webcam face detection...")
        print("Press 'q' to exit")
        processor.start_webcam_detection(show_preview=True)
    elif args.video_file:
        if os.path.exists(args.video_file):
            results = processor.process_video_file(args.video_file)
            print(f"Processed {results.get('frame_count', 0)} frames")
        else:
            print(f"Error: Video file not found: {args.video_file}")
    else:
        print("No input specified. Please use --webcam or --video-file.")
        print("Example: python face_detection_node.py --webcam")
        print("Example: python face_detection_node.py --video-file video.mp4")

if __name__ == "__main__":
    main() 