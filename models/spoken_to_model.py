import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import torch
from enum import Enum
import cv2
import mediapipe as mp
import math
import time
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """
        Initialize the Spoken-to model for laptop usage.
        
        Args:
            face_direction_threshold: Threshold for considering face as towards camera
            min_speaker_confidence: Minimum confidence for speaker detection
            min_speaker_duration: Minimum duration for speaker segment
            face_angle_threshold: Maximum angle (degrees) to consider face as towards
            speaker_timeout: Timeout in seconds for removing inactive speakers
        """
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
        
        logger.info("Initialized Spoken-to model with face detection for laptop usage")
    
    def _cleanup_inactive_speakers(self):
        """Remove speakers that haven't been seen in a while."""
        current_time = time.time()
        if current_time - self.last_cleanup_time < 1.0:  # Only cleanup once per second
            return
            
        self.last_cleanup_time = current_time
        
        # Remove inactive speakers
        inactive_speakers = []
        for speaker_id, speaker in self.active_speakers.items():
            if current_time - speaker.last_seen > self.speaker_timeout:
                inactive_speakers.append(speaker_id)
                logger.info(f"Removing inactive speaker {speaker_id}")
        
        for speaker_id in inactive_speakers:
            del self.active_speakers[speaker_id]
    
    def _update_speaker(self, speaker_id: str, face_direction: FaceDirection, 
                       face_angle: Optional[float], confidence: float):
        """Update or add a speaker to the active speakers list."""
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
            logger.info(f"Added new speaker {speaker_id}")
    
    def process_frame(self, frame: np.ndarray):
        """
        Process a video frame to detect face direction.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Tuple containing face direction, confidence, and face angle
        """
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
        """
        Calculate the angle of the face relative to the camera.
        
        Args:
            landmarks: Face mesh landmarks from MediaPipe
            
        Returns:
            float: Angle in degrees (0 = facing camera, positive = turned right, negative = turned left)
        """
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
        """
        Detect if a face is looking toward or away from the camera.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple[FaceDirection, float, Optional[float]]: Direction, confidence, and face angle (if available)
        """
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
            logger.error(f"Error detecting face direction: {str(e)}")
            return FaceDirection.UNKNOWN, 0.0, None
    
    def is_spoken_to(self, 
                    speaker_count: int,
                    active_speaker_id: Optional[str] = None,
                    face_direction: Optional[FaceDirection] = None,
                    face_confidence: float = 0.0,
                    face_angle: Optional[float] = None) -> SpokenToResult:
        """
        Determine if the user is being spoken to based on available data.
        
        Args:
            speaker_count: Number of speakers in the current segment
            active_speaker_id: ID of the active speaker, if known
            face_direction: Direction the face is pointing, if detected
            face_confidence: Confidence in the face direction detection
            face_angle: Angle of the face relative to the camera
            
        Returns:
            SpokenToResult: Result with spoken-to analysis
        """
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
            
        # If we have face direction information
        if face_direction is not None and face_confidence >= self.min_speaker_confidence:
            if face_direction == FaceDirection.TOWARDS:
                result.is_spoken_to = True
                result.confidence = face_confidence
                result.explanation = "Face is directed towards camera"
            else:
                result.is_spoken_to = False
                result.confidence = face_confidence
                result.explanation = "Face is directed away from camera"
                
            return result
            
        # If we have face angle information but not explicit direction
        if face_angle is not None:
            if abs(face_angle) < self.face_angle_threshold:
                result.is_spoken_to = True
                result.confidence = 0.7
                result.explanation = f"Face angle ({face_angle:.1f}°) indicates looking at camera"
            else:
                result.is_spoken_to = False
                result.confidence = 0.7
                result.explanation = f"Face angle ({face_angle:.1f}°) indicates looking away"
                
            return result
        
        # Fallback: with multiple speakers and no other information,
        # assume not being spoken to directly
        result.is_spoken_to = False
        result.confidence = 0.6
        result.explanation = "Multiple speakers with no face direction data"
        return result

def main():
    """Test the model with a webcam."""
    model = SpokenToModel()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        face_direction, confidence, face_angle = model.process_frame(frame)
        
        # Display result
        if face_direction != FaceDirection.UNKNOWN:
            cv2.putText(frame, f"{face_direction.value}: {confidence:.2f}, Angle: {face_angle:.1f}°",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow('Face Direction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 