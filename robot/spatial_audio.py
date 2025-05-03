import numpy as np
import librosa
from typing import Tuple, Dict, Optional, List
import logging
from dataclasses import dataclass
import math
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpeakerLocation:
    angle: float  # Azimuth angle in degrees
    confidence: float  # Confidence score (0-1)
    distance: Optional[float] = None  # Distance in meters if available
    speaker_id: Optional[str] = None  # NeMo speaker ID if available
    last_seen: float = 0.0  # Timestamp of last detection

class SpatialAudioProcessor:
    def __init__(self, sample_rate: int = 48000):
        """
        Initialize the spatial audio processor for Azure Kinect.
        
        Args:
            sample_rate (int): Sample rate of the audio (default: 48000 for Kinect)
        """
        self.sample_rate = sample_rate
        
        # Azure Kinect microphone array geometry (7 mics in a circle)
        # These are approximate values - should be calibrated for your specific setup
        self.mic_positions = np.array([
            [0.0, 0.0, 0.0],  # Center mic
            [0.042, 0.0, 0.0],  # Front
            [0.021, 0.036, 0.0],  # Front-right
            [-0.021, 0.036, 0.0],  # Back-right
            [-0.042, 0.0, 0.0],  # Back
            [-0.021, -0.036, 0.0],  # Back-left
            [0.021, -0.036, 0.0],  # Front-left
        ])
        
        # Speed of sound in air (m/s)
        self.c = 343.0
        
        # Grid search parameters for SRP-PHAT
        self.angle_grid = np.arange(-180, 180, 1)
        self.distance_grid = np.arange(0.5, 5.0, 0.1)
        
        # Confidence thresholds
        self.min_confidence = 0.3
        self.peak_to_sidelobe_ratio = 2.0
        
        # Multiple speaker tracking
        self.max_speakers = 4  # Maximum number of simultaneous speakers to track
        self.speaker_history = deque(maxlen=10)  # Keep track of last 10 frames
        self.speaker_timeout = 2.0  # Seconds before considering a speaker inactive
        
        # Depth fusion parameters
        self.depth_search_window = 5  # Degrees to search around detected angle
        self.min_depth_confidence = 0.5  # Minimum confidence for depth readings
        self.depth_fusion_window = 3  # Number of frames to average depth over

    def compute_srp_phat(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SRP-PHAT (Steered Response Power - Phase Transform) for DoA estimation.
        
        Args:
            audio_data (np.ndarray): Multi-channel audio data (channels x samples)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (angles, confidences) for all detected peaks
        """
        n_channels, n_samples = audio_data.shape
        
        # Compute cross-spectrum
        X = np.fft.rfft(audio_data, axis=1)
        G = np.zeros((n_channels, n_channels, X.shape[1]), dtype=np.complex128)
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                G[i, j] = X[i] * np.conj(X[j])
                G[j, i] = np.conj(G[i, j])
        
        # Normalize cross-spectrum (PHAT)
        G = G / (np.abs(G) + 1e-10)
        
        # Compute SRP for each angle
        srp = np.zeros(len(self.angle_grid))
        for i, angle in enumerate(self.angle_grid):
            # Steering vector for this angle
            tau = np.zeros((n_channels, n_channels))
            for m in range(n_channels):
                for n in range(m + 1, n_channels):
                    # Time delay between microphones
                    r = self.mic_positions[m] - self.mic_positions[n]
                    theta = np.radians(angle)
                    u = np.array([np.cos(theta), np.sin(theta), 0])
                    tau[m, n] = np.dot(r, u) / self.c
                    tau[n, m] = -tau[m, n]
            
            # Sum over frequency bins
            for m in range(n_channels):
                for n in range(n_channels):
                    if m != n:
                        phase = 2 * np.pi * np.arange(X.shape[1]) * tau[m, n] / n_samples
                        srp[i] += np.real(np.sum(G[m, n] * np.exp(1j * phase)))
        
        # Find multiple peaks
        peaks = []
        confidences = []
        
        # Find local maxima
        for i in range(1, len(srp)-1):
            if srp[i] > srp[i-1] and srp[i] > srp[i+1]:
                # Compute confidence for this peak
                sidelobe_mask = np.ones_like(srp, dtype=bool)
                sidelobe_mask[max(0, i-5):min(len(srp), i+6)] = False
                sidelobe_mean = np.mean(srp[sidelobe_mask])
                confidence = (srp[i] - sidelobe_mean) / (srp[i] + sidelobe_mean + 1e-10)
                confidence = min(1.0, max(0.0, confidence))
                
                if confidence >= self.min_confidence:
                    peaks.append(self.angle_grid[i])
                    confidences.append(confidence)
        
        # Sort by confidence and take top N
        if peaks:
            sorted_indices = np.argsort(confidences)[::-1]
            peaks = np.array(peaks)[sorted_indices[:self.max_speakers]]
            confidences = np.array(confidences)[sorted_indices[:self.max_speakers]]
        
        return np.array(peaks), np.array(confidences)

    def estimate_distance(self, angle: float, depth_data: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Estimate distance using depth data if available.
        
        Args:
            angle (float): Azimuth angle in degrees
            depth_data (Optional[np.ndarray]): Depth image from Kinect
            
        Returns:
            Optional[float]: Estimated distance in meters, or None if not available
        """
        if depth_data is None:
            return None
            
        # Convert angle to depth image coordinates
        center_x = depth_data.shape[1] // 2
        center_y = depth_data.shape[0] // 2
        
        # Search in a window around the detected angle
        depths = []
        confidences = []
        
        for offset in range(-self.depth_search_window, self.depth_search_window + 1):
            search_angle = angle + offset
            theta = np.radians(search_angle)
            
            # Project multiple rays at different distances
            for dist in range(50, 200, 10):
                ray_x = int(center_x + dist * np.cos(theta))
                ray_y = int(center_y + dist * np.sin(theta))
                
                if 0 <= ray_x < depth_data.shape[1] and 0 <= ray_y < depth_data.shape[0]:
                    depth = depth_data[ray_y, ray_x]
                    if depth > 0:  # Valid depth reading
                        # Compute confidence based on distance from center of search window
                        confidence = 1.0 - abs(offset) / self.depth_search_window
                        depths.append(depth)
                        confidences.append(confidence)
        
        if depths:
            # Weighted average of depth readings
            weights = np.array(confidences)
            weights = weights / np.sum(weights)
            return np.sum(np.array(depths) * weights)
                
        return None

    def update_speaker_tracking(self, current_time: float, angles: np.ndarray, 
                              confidences: np.ndarray, speaker_ids: Optional[List[str]] = None) -> List[SpeakerLocation]:
        """
        Update speaker tracking with new detections.
        
        Args:
            current_time (float): Current timestamp
            angles (np.ndarray): Detected angles
            confidences (np.ndarray): Confidence scores
            speaker_ids (Optional[List[str]]): NeMo speaker IDs if available
            
        Returns:
            List[SpeakerLocation]: Updated speaker locations
        """
        # Remove old speakers
        active_speakers = [s for s in self.speaker_history[-1] if 
                         current_time - s.last_seen < self.speaker_timeout] if self.speaker_history else []
        
        # Match new detections to existing speakers
        new_speakers = []
        for i, (angle, conf) in enumerate(zip(angles, confidences)):
            speaker_id = speaker_ids[i] if speaker_ids and i < len(speaker_ids) else None
            
            # Find closest existing speaker
            min_dist = float('inf')
            best_match = None
            
            for speaker in active_speakers:
                angle_diff = min(abs(angle - speaker.angle), 360 - abs(angle - speaker.angle))
                if angle_diff < min_dist:
                    min_dist = angle_diff
                    best_match = speaker
            
            # If close enough to existing speaker, update it
            if best_match and min_dist < 30:  # 30 degrees threshold
                best_match.angle = angle
                best_match.confidence = conf
                best_match.last_seen = current_time
                if speaker_id:
                    best_match.speaker_id = speaker_id
                new_speakers.append(best_match)
            else:
                # Create new speaker
                new_speakers.append(SpeakerLocation(
                    angle=angle,
                    confidence=conf,
                    speaker_id=speaker_id,
                    last_seen=current_time
                ))
        
        # Update history
        self.speaker_history.append(new_speakers)
        return new_speakers

    def process_audio(self, audio_data: np.ndarray, depth_data: Optional[np.ndarray] = None,
                     current_time: float = 0.0, speaker_ids: Optional[List[str]] = None) -> List[SpeakerLocation]:
        """
        Process multi-channel audio data to estimate speaker directions and distances.
        
        Args:
            audio_data (np.ndarray): Multi-channel audio data (channels x samples)
            depth_data (Optional[np.ndarray]): Depth image from Kinect
            current_time (float): Current timestamp
            speaker_ids (Optional[List[str]]): NeMo speaker IDs if available
            
        Returns:
            List[SpeakerLocation]: List of detected speaker locations
        """
        # Compute DoA using SRP-PHAT
        angles, confidences = self.compute_srp_phat(audio_data)
        
        # Update speaker tracking
        speakers = self.update_speaker_tracking(current_time, angles, confidences, speaker_ids)
        
        # Estimate distances for each speaker
        for speaker in speakers:
            speaker.distance = self.estimate_distance(speaker.angle, depth_data)
        
        return speakers

def get_speaker_direction(audio_file: str, depth_data: Optional[np.ndarray] = None,
                         speaker_ids: Optional[List[str]] = None) -> List[Dict[str, float]]:
    """
    Process spatial audio to determine speaker directions using Azure Kinect microphone array.
    
    Args:
        audio_file (str): Path to the audio file
        depth_data (Optional[np.ndarray]): Depth image from Kinect
        speaker_ids (Optional[List[str]]): NeMo speaker IDs if available
        
    Returns:
        List[Dict[str, float]]: List of dictionaries containing speaker information:
            - 'angle': Angle in degrees (0 is front, 90 is right, -90 is left)
            - 'confidence': Confidence score between 0 and 1
            - 'distance': Estimated distance in meters if available
            - 'speaker_id': NeMo speaker ID if available
    """
    # Load the audio file
    audio_data, sr = librosa.load(audio_file, sr=None, mono=False)
    
    # Initialize processor
    processor = SpatialAudioProcessor(sr)
    
    # Process audio
    speakers = processor.process_audio(audio_data, depth_data, speaker_ids=speaker_ids)
    
    # Convert to dictionary format
    return [{
        'angle': speaker.angle,
        'confidence': speaker.confidence,
        'distance': speaker.distance if speaker.distance is not None else 0.0,
        'speaker_id': speaker.speaker_id
    } for speaker in speakers]

# Example usage:
if __name__ == "__main__":
    audio_file = "data/audio/sample_stereo_audio.wav"
    results = get_speaker_direction(audio_file)
    
    for result in results:
        print(f"\nSpeaker {result.get('speaker_id', 'Unknown')}:")
        print(f"Angle: {result['angle']:.1f}° (confidence: {result['confidence']:.2f})")
        if result['distance'] > 0:
            print(f"Distance: {result['distance']:.2f}m")
