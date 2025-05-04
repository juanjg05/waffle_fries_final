import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import torch
from enum import Enum
import cv2
import mediapipe as mp
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import cv_bridge
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

class SpokenToModel(Node):
    def __init__(self, 
                 face_direction_threshold: float = 0.7,  # Threshold for considering face as towards robot
                 min_speaker_confidence: float = 0.6,    # Minimum confidence for speaker detection
                 min_speaker_duration: float = 0.5,      # Minimum duration for speaker segment
                 face_angle_threshold: float = 30.0,     # Maximum angle (degrees) to consider face as towards
                 speaker_timeout: float = 60.0):         # Timeout in seconds for removing inactive speakers
        """
        Initialize the Spoken-to model.
        
        Args:
            face_direction_threshold: Threshold for considering face as towards robot
            min_speaker_confidence: Minimum confidence for speaker detection
            min_speaker_duration: Minimum duration for speaker segment
            face_angle_threshold: Maximum angle (degrees) to consider face as towards
            speaker_timeout: Timeout in seconds for removing inactive speakers
        """
        super().__init__('spoken_to_model')
        
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
        
        # Initialize ROS2 components
        self.bridge = cv_bridge.CvBridge()
        
        # Subscribe to Azure Kinect camera
        self.camera_sub = self.create_subscription(
            Image,
            '/k4a/rgb/image_raw',  # Azure Kinect RGB topic
            self.camera_callback,
            10
        )
        
        # Subscribe to audio direction
        self.audio_sub = self.create_subscription(
            Float32,
            '/k4a/audio_direction',  # Azure Kinect audio direction topic
            self.audio_callback,
            10
        )
        
        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',  # Standard ROS2 cmd_vel topic
            10
        )
        
        # Store latest audio direction
        self.latest_audio_angle = None
        self.audio_confidence = 0.0
        
        # Track active speakers
        self.active_speakers = {}  # Dict[str, Speaker]
        self.last_cleanup_time = time.time()
        
        logger.info("Initialized Spoken-to model with face and audio detection")
    
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
    
    def audio_callback(self, msg: Float32):
        """
        Process incoming audio direction messages.
        
        Args:
            msg: ROS2 Float32 message containing audio direction angle
        """
        try:
            # Store the latest audio direction
            self.latest_audio_angle = msg.data
            self.audio_confidence = 0.9  # High confidence for audio direction
            
            # If we have a valid audio direction, turn towards it
            if self.latest_audio_angle is not None:
                self.turn_towards_speaker(self.latest_audio_angle)
            
        except Exception as e:
            logger.error(f"Error in audio callback: {str(e)}")
    
    def camera_callback(self, msg: Image):
        """
        Process incoming camera images.
        
        Args:
            msg: ROS2 Image message from Azure Kinect
        """
        try:
            # Convert ROS2 image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect face direction
            face_direction, confidence, face_angle = self.detect_face_direction(cv_image)
            
            # Update speaker tracking
            if face_direction != FaceDirection.UNKNOWN:
                # For now, use a simple ID based on face position
                speaker_id = f"speaker_{int(face_angle) if face_angle is not None else 0}"
                self._update_speaker(speaker_id, face_direction, face_angle, confidence)
            
            # Clean up inactive speakers
            self._cleanup_inactive_speakers()
            
            # If face is detected and at an angle, and we don't have audio direction,
            # use face direction to turn
            if (face_direction == FaceDirection.TOWARDS and 
                face_angle is not None and 
                self.latest_audio_angle is None):
                if abs(face_angle) > 5.0:  # 5 degree threshold for turning
                    self.turn_towards_speaker(face_angle)
            
        except Exception as e:
            logger.error(f"Error in camera callback: {str(e)}")
    
    def _calculate_face_angle(self, landmarks) -> float:
        """
        Calculate the angle of the face relative to the camera.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            Angle in degrees (0 = facing camera, positive = turned right, negative = turned left)
        """
        try:
            # Get nose tip and face center points
            nose_tip = landmarks[1]  # Nose tip landmark
            left_eye = landmarks[33]  # Left eye landmark
            right_eye = landmarks[263]  # Right eye landmark
            
            # Calculate face center
            face_center_x = (left_eye.x + right_eye.x) / 2
            face_center_y = (left_eye.y + right_eye.y) / 2
            
            # Calculate angle between nose tip and face center
            dx = nose_tip.x - face_center_x
            dy = nose_tip.y - face_center_y
            angle = math.degrees(math.atan2(dy, dx))
            
            return angle
        except Exception as e:
            logger.error(f"Error calculating face angle: {str(e)}")
            return 0.0
    
    def detect_face_direction(self, image: np.ndarray) -> Tuple[FaceDirection, float, Optional[float]]:
        """
        Detect if a face is facing towards or away from the robot.
        
        Args:
            image: RGB image containing the face
            
        Returns:
            Tuple of (FaceDirection, confidence, face_angle)
        """
        try:
            # Use MediaPipe for face detection
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return FaceDirection.UNKNOWN, 0.0, None
            
            # Get landmarks for the first face
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate face angle
            face_angle = self._calculate_face_angle(landmarks)
            
            # Determine face direction based on angle
            if abs(face_angle) <= self.face_angle_threshold:
                return FaceDirection.TOWARDS, 0.9, face_angle
            else:
                return FaceDirection.AWAY, 0.9, face_angle
                
        except Exception as e:
            logger.error(f"Error in face direction detection: {str(e)}")
            return FaceDirection.UNKNOWN, 0.0, None
    
    def turn_towards_speaker(self, angle: float) -> bool:
        """
        Turn the robot towards the speaker.
        
        Args:
            angle: Angle of the speaker relative to the robot (degrees)
            
        Returns:
            True if turn was successful, False otherwise
        """
        try:
            # Create Twist message for robot movement
            cmd = Twist()
            
            # Set angular velocity based on angle
            # Negative angle means turn left, positive means turn right
            cmd.angular.z = -angle * 0.1  # Scale factor for smooth turning
            
            # Publish command
            self.cmd_vel_pub.publish(cmd)
            
            logger.info(f"Turning to angle {angle} degrees")
            return True
            
        except Exception as e:
            logger.error(f"Error turning towards speaker: {str(e)}")
            return False
    
    def is_spoken_to(self, 
                    speaker_count: int,
                    active_speaker_id: Optional[str] = None,
                    face_direction: Optional[FaceDirection] = None,
                    face_confidence: float = 0.0,
                    face_angle: Optional[float] = None) -> SpokenToResult:
        """
        Determine if the robot is being spoken to based on speaker count and face direction.
        
        Args:
            speaker_count: Number of speakers detected
            active_speaker_id: ID of the currently active speaker
            face_direction: Direction the face is pointing
            face_confidence: Confidence of face direction detection
            face_angle: Angle of the face relative to the camera
            
        Returns:
            SpokenToResult with decision and explanation
        """
        try:
            # If only one speaker, they are likely speaking to the robot
            if speaker_count == 1:
                return SpokenToResult(
                    is_spoken_to=True,
                    confidence=0.9,
                    face_direction=face_direction or FaceDirection.UNKNOWN,
                    speaker_count=speaker_count,
                    active_speaker_id=active_speaker_id,
                    explanation="Single speaker detected - likely speaking to robot",
                    face_angle=face_angle
                )
            
            # If multiple speakers, check face direction
            if face_direction == FaceDirection.TOWARDS and face_confidence >= self.face_direction_threshold:
                # If face is towards but at an angle, try to turn towards speaker
                if face_angle is not None and abs(face_angle) > 5.0:  # 5 degree threshold for turning
                    self.turn_towards_speaker(face_angle)
                
                return SpokenToResult(
                    is_spoken_to=True,
                    confidence=face_confidence,
                    face_direction=face_direction,
                    speaker_count=speaker_count,
                    active_speaker_id=active_speaker_id,
                    explanation="Face is directed towards robot",
                    face_angle=face_angle
                )
            elif face_direction == FaceDirection.AWAY:
                return SpokenToResult(
                    is_spoken_to=False,
                    confidence=face_confidence,
                    face_direction=face_direction,
                    speaker_count=speaker_count,
                    active_speaker_id=active_speaker_id,
                    explanation="Face is directed away from robot",
                    face_angle=face_angle
                )
            else:
                # If face direction is unknown or confidence is low
                return SpokenToResult(
                    is_spoken_to=False,
                    confidence=0.5,
                    face_direction=face_direction or FaceDirection.UNKNOWN,
                    speaker_count=speaker_count,
                    active_speaker_id=active_speaker_id,
                    explanation="Cannot determine if robot is being spoken to",
                    face_angle=face_angle
                )
                
        except Exception as e:
            logger.error(f"Error in is_spoken_to: {str(e)}")
            return SpokenToResult(
                is_spoken_to=False,
                confidence=0.0,
                face_direction=FaceDirection.UNKNOWN,
                speaker_count=speaker_count,
                active_speaker_id=active_speaker_id,
                explanation=f"Error: {str(e)}",
                face_angle=face_angle
            )

def main(args=None):
    rclpy.init(args=args)
    node = SpokenToModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main() 