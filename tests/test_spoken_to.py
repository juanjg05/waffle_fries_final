import pytest
import numpy as np
import cv2
from models.spoken_to_model import SpokenToModel, FaceDirection, SpokenToResult

def create_test_face_image(angle: float = 0.0) -> np.ndarray:
    """Create a test image with a face at a specific angle."""
    # Create a blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple face (circle for head, dots for eyes)
    center = (320, 240)
    cv2.circle(image, center, 100, (255, 255, 255), -1)  # Head
    
    # Calculate eye positions based on angle
    eye_distance = 40
    eye_y = 220
    eye_x_offset = int(eye_distance * np.sin(np.radians(angle)))
    
    # Draw eyes
    cv2.circle(image, (center[0] - eye_x_offset, eye_y), 10, (0, 0, 0), -1)
    cv2.circle(image, (center[0] + eye_x_offset, eye_y), 10, (0, 0, 0), -1)
    
    return image

def test_single_speaker():
    """Test that a single speaker is considered as speaking to the robot."""
    model = SpokenToModel()
    result = model.is_spoken_to(
        speaker_count=1,
        active_speaker_id="SPEAKER_01"
    )
    
    assert result.is_spoken_to == True
    assert result.confidence >= 0.9
    assert result.speaker_count == 1
    assert result.active_speaker_id == "SPEAKER_01"
    assert "Single speaker" in result.explanation

def test_face_direction_detection():
    """Test face direction detection with different angles."""
    model = SpokenToModel()
    
    # Test face looking straight
    image_straight = create_test_face_image(0.0)
    direction, confidence, angle = model.detect_face_direction(image_straight)
    assert direction == FaceDirection.TOWARDS
    assert confidence >= 0.9
    assert abs(angle) <= model.face_angle_threshold
    
    # Test face looking away
    image_away = create_test_face_image(45.0)
    direction, confidence, angle = model.detect_face_direction(image_away)
    assert direction == FaceDirection.AWAY
    assert confidence >= 0.9
    assert abs(angle) > model.face_angle_threshold

def test_turn_towards_speaker():
    """Test turning towards speaker functionality."""
    model = SpokenToModel()
    
    # Test turning to a specific angle
    success = model.turn_towards_speaker(30.0)
    assert isinstance(success, bool)  # We can't assert True/False as it depends on Kinect availability

def test_multiple_speakers_face_towards():
    """Test that a speaker facing the robot is considered as speaking to it."""
    model = SpokenToModel()
    
    # Create test image with face looking towards
    image = create_test_face_image(0.0)
    direction, confidence, angle = model.detect_face_direction(image)
    
    result = model.is_spoken_to(
        speaker_count=2,
        active_speaker_id="SPEAKER_01",
        face_direction=direction,
        face_confidence=confidence,
        face_angle=angle
    )
    
    assert result.is_spoken_to == True
    assert result.confidence >= 0.8
    assert result.speaker_count == 2
    assert result.face_direction == FaceDirection.TOWARDS
    assert "Face is directed towards" in result.explanation
    assert result.face_angle is not None

def test_multiple_speakers_face_away():
    """Test that a speaker facing away is not considered as speaking to the robot."""
    model = SpokenToModel()
    
    # Create test image with face looking away
    image = create_test_face_image(45.0)
    direction, confidence, angle = model.detect_face_direction(image)
    
    result = model.is_spoken_to(
        speaker_count=2,
        active_speaker_id="SPEAKER_01",
        face_direction=direction,
        face_confidence=confidence,
        face_angle=angle
    )
    
    assert result.is_spoken_to == False
    assert result.confidence >= 0.8
    assert result.speaker_count == 2
    assert result.face_direction == FaceDirection.AWAY
    assert "Face is directed away" in result.explanation
    assert result.face_angle is not None

def test_multiple_speakers_unknown_face():
    """Test that unknown face direction results in not speaking to robot."""
    model = SpokenToModel()
    result = model.is_spoken_to(
        speaker_count=2,
        active_speaker_id="SPEAKER_01",
        face_direction=FaceDirection.UNKNOWN,
        face_confidence=0.5
    )
    
    assert result.is_spoken_to == False
    assert result.confidence == 0.5
    assert result.speaker_count == 2
    assert result.face_direction == FaceDirection.UNKNOWN
    assert "Cannot determine" in result.explanation

def test_process_audio_segment():
    """Test processing an audio segment with face detection."""
    model = SpokenToModel()
    
    # Create test image with face
    image = create_test_face_image(0.0)
    
    result = model.process_audio_segment(
        speaker_id="SPEAKER_01",
        start_time=0.0,
        end_time=1.0,
        confidence=0.8,
        image=image
    )
    
    assert isinstance(result, SpokenToResult)
    assert result.active_speaker_id == "SPEAKER_01"
    assert result.speaker_count == 1
    assert result.face_angle is not None

def test_short_segment():
    """Test that very short segments are rejected."""
    model = SpokenToModel(min_speaker_duration=0.5)
    result = model.process_audio_segment(
        speaker_id="SPEAKER_01",
        start_time=0.0,
        end_time=0.2,  # Too short
        confidence=0.8
    )
    
    assert result.is_spoken_to == False
    assert "Segment too short" in result.explanation

def test_low_confidence():
    """Test that segments with low confidence are rejected."""
    model = SpokenToModel(min_speaker_confidence=0.6)
    result = model.process_audio_segment(
        speaker_id="SPEAKER_01",
        start_time=0.0,
        end_time=1.0,
        confidence=0.4  # Too low
    )
    
    assert result.is_spoken_to == False
    assert "Low speaker confidence" in result.explanation 