import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from enum import Enum
import cv2
import mediapipe as mp
import math
import time
from collections import defaultdict

class FaceDirection(Enum):
    TOWARDS = "towards"
    AWAY = "away"
    UNKNOWN = "unknown"

@dataclass
class Speaker:
    id: str
    last_seen: float
    face_direction: FaceDirection
    face_angle: Optional[float]
    confidence: float

@dataclass
class SpokenToResult:
    is_spoken_to: bool
    confidence: float
    face_direction: FaceDirection
    speaker_count: int
    active_speaker_id: Optional[str] = None
    explanation: str = ""
    face_angle: Optional[float] = None  # Angle of face relative to camera
    audio_angle: Optional[float] = None  # Angle of audio source

class SpokenToModel:
    def __init__(self, 
                 face_direction_threshold: float = 0.7,  # Threshold for considering face as towards camera
                 min_speaker_confidence: float = 0.6,    # Minimum confidence for speaker detection
                 min_speaker_duration: float = 0.5,      # Minimum duration for speaker segment
                 face_angle_threshold: float = 30.0,     # Maximum angle (degrees) to consider face as towards
                 speaker_timeout: float = 60.0):         # Timeout in seconds for removing inactive speakers
        self.face_direction_threshold = face_direction_threshold
        self.min_speaker_confidence = min_speaker_confidence
        self.min_speaker_duration = min_speaker_duration
        self.face_angle_threshold = face_angle_threshold
        self.speaker_timeout = speaker_timeout
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Store latest audio direction
        self.latest_audio_angle = None
        self.audio_confidence = 0.0
        
        # Track active speakers
        self.active_speakers = {}  # Dict[str, Speaker]
        self.last_cleanup_time = time.time()
    
    def _cleanup_inactive_speakers(self):
        current_time = time.time()
        if current_time - self.last_cleanup_time < 1.0:  # Only cleanup once per second
            return
            
        self.last_cleanup_time = current_time
        
        # Remove inactive speakers
        inactive_speakers = []
        for speaker_id, speaker in self.active_speakers.items():
            if current_time - speaker.last_seen > self.speaker_timeout:
                inactive_speakers.append(speaker_id)
        
        for speaker_id in inactive_speakers:
            del self.active_speakers[speaker_id]
    
    def _update_speaker(self, speaker_id: str, face_direction: FaceDirection, 
                       face_angle: Optional[float], confidence: float):
        current_time = time.time()
        
        if speaker_id in self.active_speakers:
            # Update existing speaker
            speaker = self.active_speakers[speaker_id]
            speaker.last_seen = current_time
            speaker.face_direction = face_direction
            speaker.face_angle = face_angle
            speaker.confidence = confidence
        else:
            # Add new speaker
            self.active_speakers[speaker_id] = Speaker(
                id=speaker_id,
                last_seen=current_time,
                face_direction=face_direction,
                face_angle=face_angle,
                confidence=confidence
            )
    
    def process_frame(self, frame: np.ndarray):
        face_direction, confidence, face_angle = self.detect_face_direction(frame)
        
        # Update speaker tracking
        if face_direction != FaceDirection.UNKNOWN:
            # For now, use a simple ID based on face position
            speaker_id = f"speaker_{int(face_angle) if face_angle is not None else 0}"
            self._update_speaker(speaker_id, face_direction, face_angle, confidence)
        
        # Clean up inactive speakers
        self._cleanup_inactive_speakers()
        
        return face_direction, confidence, face_angle
    
    def _calculate_face_angle(self, landmarks) -> float:
        # Get nose point
        nose_point = landmarks[4]
        
        # Get points for left and right sides of face
        left_cheek = landmarks[234]  # Left cheek
        right_cheek = landmarks[454]  # Right cheek
        
        # Calculate face center point and direction vector
        face_center_x = (left_cheek.x + right_cheek.x) / 2
        face_center_y = (left_cheek.y + right_cheek.y) / 2
        
        # Calculate face direction vector
        direction_vector = (nose_point.x - face_center_x, nose_point.y - face_center_y)
        
        # Calculate angle (in radians)
        angle_rad = math.atan2(direction_vector[0], direction_vector[1])
        
        # Convert to degrees and normalize
        angle_deg = angle_rad * 180 / math.pi
        
        return angle_deg
    
    def detect_face_direction(self, image: np.ndarray) -> Tuple[FaceDirection, float, Optional[float]]:
        try:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return FaceDirection.UNKNOWN, 0.0, None
                
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate face angle
            face_angle = self._calculate_face_angle(face_landmarks)
            
            # Determine face direction based on angle
            if abs(face_angle) < self.face_angle_threshold:
                return FaceDirection.TOWARDS, 0.9, face_angle
            else:
                return FaceDirection.AWAY, 0.8, face_angle
                
        except Exception as e:
            print(f"Error detecting face direction: {str(e)}")
            return FaceDirection.UNKNOWN, 0.0, None
    
    def is_spoken_to(self, 
                    speaker_count: int,
                    active_speaker_id: Optional[str] = None,
                    face_direction: Optional[FaceDirection] = None,
                    face_confidence: float = 0.0,
                    face_angle: Optional[float] = None) -> SpokenToResult:
        # Default result
        result = SpokenToResult(
            is_spoken_to=False,
            confidence=0.5,
            face_direction=FaceDirection.UNKNOWN if face_direction is None else face_direction,
            speaker_count=speaker_count,
            active_speaker_id=active_speaker_id,
            face_angle=face_angle,
            audio_angle=self.latest_audio_angle
        )
        
        # Simple case: if we have only one speaker, they're likely speaking to the user
        if speaker_count == 1:
            result.is_spoken_to = True
            result.confidence = 0.8
            result.explanation = "Only one speaker detected"
            return result
        
        # If we have face direction information, use it
        if face_direction is not None and face_confidence > self.face_direction_threshold:
            if face_direction == FaceDirection.TOWARDS:
                result.is_spoken_to = True
                result.confidence = face_confidence
                result.explanation = "Face is pointing towards camera"
            else:
                result.is_spoken_to = False
                result.confidence = face_confidence
                result.explanation = "Face is pointing away from camera"
            
            return result
        
        # Fallback: if we have multiple speakers, assume not spoken to
        result.is_spoken_to = False
        result.confidence = 0.6
        result.explanation = f"Multiple speakers ({speaker_count}) detected without clear face direction"
        
        return result
    
    def set_audio_direction(self, angle: Optional[float], confidence: float = 0.0):
        """Set the estimated direction of the audio source relative to the camera."""
        self.latest_audio_angle = angle
        self.audio_confidence = confidence
    
    def get_active_speakers(self) -> List[Speaker]:
        """Return list of currently active speakers."""
        return list(self.active_speakers.values())
    
    def get_speaker_status(self, speaker_id: str) -> Optional[Dict]:
        """Get status of a specific speaker."""
        if speaker_id not in self.active_speakers:
            return None
            
        speaker = self.active_speakers[speaker_id]
        return {
            'id': speaker.id,
            'face_direction': speaker.face_direction.value,
            'face_angle': speaker.face_angle,
            'confidence': speaker.confidence,
            'seconds_since_last_seen': time.time() - speaker.last_seen
        }

def main():
    import argparse
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Spoken-to model demo")
    parser.add_argument("--video", help="Path to video file to process")
    args = parser.parse_args()
    
    # create model
    model = SpokenToModel()
    
    # open video capture
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Processing video... Press 'q' to quit.")
    
    # process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # process frame
        face_direction, confidence, face_angle = model.process_frame(frame)
        
        # status
        result = model.is_spoken_to(
            speaker_count=2,
            face_direction=face_direction,
            face_confidence=confidence,
            face_angle=face_angle
        )
        
    
        direction_text = f"Direction: {face_direction.value}"
        angle_text = f"Angle: {face_angle:.1f}°" if face_angle is not None else "Angle: unknown"
        spoken_to_text = f"Spoken To: {'Yes' if result.is_spoken_to else 'No'} ({result.confidence:.2f})"
        
        # show frame
        color = (0, 255, 0) if result.is_spoken_to else (0, 0, 255)
        cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, angle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, spoken_to_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # display frame
        cv2.imshow("Spoken-to Model Demo", frame)
        
        # check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main() 