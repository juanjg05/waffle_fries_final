#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Any, Dict
import json
import os
import time
from datetime import datetime

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_face_direction(self, frame: np.ndarray) -> Dict:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'direction': 'unknown',
                'angle': None,
                'confidence': 0.0,
                'is_spoken_to': False
            }
            
        face_landmarks = results.multi_face_landmarks[0]
        
        nose_bridge = [
            face_landmarks.landmark[1],
            face_landmarks.landmark[4]
        ]
        
        dx = nose_bridge[1].x - nose_bridge[0].x
        dy = nose_bridge[1].y - nose_bridge[0].y
        angle = np.degrees(np.arctan2(dy, dx))
        
        if abs(angle) < 30:
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
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.face_detector = FaceDetector()
        
        self.cap = None
        self.video_width = 640
        self.video_height = 480
        self.fps = 30
        
    def process_video_file(self, video_file: str) -> Dict:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            return {'error': 'Failed to open video file'}
        
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        face_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / self.fps
            face_info = self.face_detector.detect_face_direction(frame)
            
            result = {
                'frame': frame_count,
                'timestamp': timestamp,
                **face_info
            }
            
            face_results.append(result)
            frame_count += 1
        
        cap.release()
        
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
        
        return {
            'filename': os.path.basename(video_file),
            'frame_count': frame_count,
            'fps': self.fps,
            'resolution': f"{self.video_width}x{self.video_height}",
            'face_results_count': len(face_results)
        }
    
    def start_webcam_detection(self, show_preview=True):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error opening webcam")
            return False
        
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        face_results = []
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = time.time() - start_time
            face_info = self.face_detector.detect_face_direction(frame)
            
            result = {
                'timestamp': timestamp,
                **face_info
            }
            
            face_results.append(result)
            
            if show_preview:
                direction = face_info['direction']
                angle = face_info['angle']
                
                color = (0, 255, 0) if direction == 'towards' else (0, 0, 255)
                text = f"Direction: {direction}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                if angle is not None:
                    text = f"Angle: {angle:.1f}°"
                    cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.imshow('Face Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        if face_results:
            result_filename = os.path.join(
                self.output_dir, 
                f"webcam_face_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(result_filename, 'w') as f:
                json.dump({
                    'source': 'webcam',
                    'processed_at': datetime.now().isoformat(),
                    'resolution': f"{self.video_width}x{self.video_height}",
                    'face_results': face_results
                }, f, indent=2)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        return True
            
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Face detection processor")
    parser.add_argument("--video-file", type=str, help="Path to video file to process")
    parser.add_argument("--output-dir", type=str, default="face_detection_results", 
                        help="Directory to save detection results")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for live detection")
    
    args = parser.parse_args()
    
    processor = FaceDetectionProcessor(output_dir=args.output_dir)
    
    if args.webcam:
        print("Starting webcam face detection. Press 'q' to quit.")
        processor.start_webcam_detection(show_preview=True)
    elif args.video_file:
        if os.path.exists(args.video_file):
            print(f"Processing video file: {args.video_file}")
            result = processor.process_video_file(args.video_file)
            print(f"Processing complete: {result}")
        else:
            print(f"Error: Video file not found: {args.video_file}")
    else:
        print("No input specified. Please provide a video file or use --webcam.")
        print("Example: python face_detection_node.py --video-file video.mp4")
        print("Example: python face_detection_node.py --webcam")

if __name__ == "__main__":
    main() 