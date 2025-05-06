#!/usr/bin/env python3

import cv2
import numpy as np
import pyaudio
import time
import threading
import whisper
import torch
import json
import os
import mediapipe as mp
from datetime import datetime
from queue import Queue
from typing import List, Dict, Optional, Tuple
import logging
from scipy.spatial.distance import cosine
import soundfile as sf
import librosa
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiarizationResult:
    speaker_id: str
    start_time: float
    end_time: float
    transcript: str = ""
    confidence: float = 1.0
    speaker_embedding: Optional[np.ndarray] = None
    speaker_confidence: float = 1.0
    is_spoken_to: bool = False
    face_angle: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert the result to a dictionary for JSON serialization."""
        return {
            "speaker_id": self.speaker_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "transcript": self.transcript,
            "diarization_confidence": self.confidence,
            "speaker_confidence": self.speaker_confidence,
            "speaker_embedding": self.speaker_embedding.tolist() if self.speaker_embedding is not None else None,
            "is_spoken_to": self.is_spoken_to,
            "face_angle": self.face_angle
        }

class SpeakerContextManager:
    def __init__(self, storage_dir="speaker_contexts"):
        """Initialize speaker context manager with storage directory."""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.speaker_data = {}
        self._load_existing_contexts()
    
    def _load_existing_contexts(self):
        """Load existing speaker contexts from storage directory."""
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.storage_dir, filename), 'r') as f:
                        speaker_data = json.load(f)
                        self.speaker_data[speaker_data.get('speaker_id')] = speaker_data
                except Exception as e:
                    logger.error(f"Failed to load speaker context {filename}: {e}")
    
    def update_speaker(self, speaker_id: str, embedding: np.ndarray = None, 
                      transcript: str = None, is_spoken_to: bool = None):
        """Update speaker context with new data."""
        if speaker_id not in self.speaker_data:
            self.speaker_data[speaker_id] = {
                'speaker_id': speaker_id,
                'first_seen': datetime.now().isoformat(),
                'transcripts': [],
                'embedding': None,
                'spoken_to_count': 0,
                'not_spoken_to_count': 0
            }
        
        speaker = self.speaker_data[speaker_id]
        speaker['last_seen'] = datetime.now().isoformat()
        
        if embedding is not None:
            speaker['embedding'] = embedding.tolist()
        
        if transcript is not None:
            speaker['transcripts'].append({
                'text': transcript,
                'timestamp': datetime.now().isoformat()
            })
            speaker['transcripts'] = speaker['transcripts'][-50:]  # Keep last 50 transcripts
        
        if is_spoken_to is not None:
            if is_spoken_to:
                speaker['spoken_to_count'] += 1
            else:
                speaker['not_spoken_to_count'] += 1
        
        # Save to file
        self._save_speaker(speaker_id)
    
    def _save_speaker(self, speaker_id: str):
        """Save speaker context to file."""
        with open(os.path.join(self.storage_dir, f"{speaker_id}.json"), 'w') as f:
            json.dump(self.speaker_data[speaker_id], f, indent=2)
    
    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """Get speaker context by ID."""
        return self.speaker_data.get(speaker_id)
    
    def get_all_speakers(self) -> List[Dict]:
        """Get all speaker contexts."""
        return list(self.speaker_data.values())

class SimplifiedLaptopProcessor:
    def __init__(self, 
                 output_dir="conversation_data",
                 model_dir="models",
                 min_speaker_duration=0.5,
                 similarity_threshold=0.85):
        """
        Initialize the laptop audio and video processor (simplified version without pyannote).
        
        Args:
            output_dir: Directory to save conversation data
            model_dir: Directory to store models
            min_speaker_duration: Minimum duration for a speaker segment
            similarity_threshold: Threshold for speaker similarity matching
        """
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize context manager
        self.context_manager = SpeakerContextManager()
        
        # Audio parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.min_speaker_duration = min_speaker_duration
        self.similarity_threshold = similarity_threshold
        
        # Initialize audio recorder
        self.pyaudio = pyaudio.PyAudio()
        self.audio_frames = []
        self.recording = False
        self.audio_thread = None
        
        # Initialize video capture
        self.cap = None
        self.video_frames = []
        self.recording_video = False
        self.video_thread = None
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Processing variables
        self.conversation_id = None
        self.conversation_data = []
        self.current_speakers = {}
        
        # Load whisper model for transcription
        logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded")
        
        logger.info(f"Simplified Laptop Processor initialized (without diarization)")

    def start_recording(self):
        """Start recording audio and video."""
        if self.recording:
            logger.warning("Already recording")
            return
        
        # Generate conversation ID
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting recording for conversation {self.conversation_id}")
        
        # Reset frames
        self.audio_frames = []
        self.video_frames = []
        
        # Start recording audio
        self.recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        # Start recording video
        self.recording_video = True
        self.cap = cv2.VideoCapture(0)
        self.video_thread = threading.Thread(target=self._record_video)
        self.video_thread.start()
        
        logger.info("Recording started. Press Enter to stop.")

    def stop_recording(self):
        """Stop recording audio and video and process the conversation."""
        if not self.recording:
            logger.warning("Not recording")
            return
        
        logger.info("Stopping recording...")
        
        # Stop recording audio
        self.recording = False
        if self.audio_thread:
            self.audio_thread.join()
        
        # Stop recording video
        self.recording_video = False
        if self.video_thread:
            self.video_thread.join()
        
        if self.cap:
            self.cap.release()
        
        logger.info("Recording stopped, processing conversation...")
        
        # Process the conversation
        self._process_conversation()

    def _record_audio(self):
        """Record audio in a separate thread."""
        try:
            stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            while self.recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_frames.append(data)
            
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            self.recording = False

    def _record_video(self):
        """Record video in a separate thread."""
        try:
            while self.recording_video:
                ret, frame = self.cap.read()
                if ret:
                    timestamp = time.time()
                    face_data = self._detect_face_direction(frame)
                    self.video_frames.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'face_data': face_data
                    })
                time.sleep(0.03)  # ~30 fps
        except Exception as e:
            logger.error(f"Error recording video: {str(e)}")
            self.recording_video = False

    def _detect_face_direction(self, frame):
        """
        Detect face direction in a frame.
        
        Args:
            frame: Video frame
            
        Returns:
            Dict with face information or None if no face detected
        """
        try:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return None
                
            # Get first face
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Get nose point
            nose_point = face_landmarks[4]
            
            # Get points for left and right sides of face
            left_cheek = face_landmarks[234]  # Left cheek
            right_cheek = face_landmarks[454]  # Right cheek
            
            # Calculate face center point and direction vector
            face_center_x = (left_cheek.x + right_cheek.x) / 2
            face_center_y = (left_cheek.y + right_cheek.y) / 2
            
            # Calculate face direction vector
            direction_vector = (nose_point.x - face_center_x, nose_point.y - face_center_y)
            
            # Calculate angle (in radians)
            angle_rad = np.arctan2(direction_vector[0], direction_vector[1])
            
            # Convert to degrees
            angle_deg = angle_rad * 180 / np.pi
            
            # Determine if facing camera (within threshold)
            is_facing_camera = abs(angle_deg) < 30.0
            
            return {
                'angle': angle_deg,
                'is_facing_camera': is_facing_camera,
                'confidence': 0.9 if is_facing_camera else 0.7
            }
                
        except Exception as e:
            logger.error(f"Error detecting face direction: {str(e)}")
            return None

    def _process_conversation(self):
        """Process the recorded conversation."""
        try:
            # Save audio
            audio_filename = os.path.join(self.output_dir, f"{self.conversation_id}_audio.wav")
            self._save_audio(audio_filename)
            
            # Save sample frames
            frames_dir = os.path.join(self.output_dir, f"{self.conversation_id}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            self._save_sample_frames(frames_dir)
            
            # Transcribe the audio
            logger.info("Transcribing audio...")
            transcription_results = self._transcribe_audio(audio_filename)
            
            # Add face information
            self._add_face_information(transcription_results)
            
            # Save results
            self._save_results(transcription_results)
            
            logger.info(f"Processing complete. Results saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")

    def _save_audio(self, filename):
        """
        Save recorded audio to a file.
        
        Args:
            filename: Output filename
        """
        try:
            # Convert audio frames to numpy array
            audio_data = b''.join(self.audio_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save as WAV file
            sf.write(filename, audio_np, self.sample_rate)
            logger.info(f"Audio saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")

    def _save_sample_frames(self, frames_dir):
        """
        Save sample frames from the video.
        
        Args:
            frames_dir: Directory to save frames
        """
        try:
            # Save a frame every second
            frame_count = len(self.video_frames)
            if frame_count == 0:
                logger.warning("No frames to save")
                return
                
            # Save every ~30th frame (assuming ~30fps)
            for i in range(0, frame_count, 30):
                if i < frame_count:
                    frame_data = self.video_frames[i]
                    frame = frame_data['frame']
                    timestamp = frame_data['timestamp']
                    
                    # Format timestamp as seconds from start
                    start_time = self.video_frames[0]['timestamp'] if self.video_frames else timestamp
                    seconds = timestamp - start_time
                    
                    # Save frame
                    filename = os.path.join(frames_dir, f"frame_{seconds:.1f}.jpg")
                    cv2.imwrite(filename, frame)
            
            logger.info(f"Saved {len(os.listdir(frames_dir))} sample frames")
        except Exception as e:
            logger.error(f"Error saving frames: {str(e)}")

    def _transcribe_audio(self, audio_filename) -> List[DiarizationResult]:
        """
        Transcribe the audio and create segments.
        Since we don't have diarization, we'll create a single speaker.
        
        Args:
            audio_filename: Path to the audio file
            
        Returns:
            List of DiarizationResult objects
        """
        try:
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(audio_filename, language="en")
            
            # Create segments from whisper segments
            segments = []
            
            for i, segment in enumerate(result["segments"]):
                # Create a diarization result for each segment
                diarization_result = DiarizationResult(
                    speaker_id=f"speaker_0",  # Single speaker without diarization
                    start_time=segment["start"],
                    end_time=segment["end"],
                    transcript=segment["text"],
                    confidence=segment.get("confidence", 0.9),
                )
                segments.append(diarization_result)
                
                # Update speaker context
                self.context_manager.update_speaker(
                    speaker_id=diarization_result.speaker_id,
                    transcript=diarization_result.transcript
                )
            
            logger.info(f"Transcribed {len(segments)} segments")
            return segments
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return []

    def _add_face_information(self, segments: List[DiarizationResult]):
        """
        Add face information to the segments.
        
        Args:
            segments: List of DiarizationResult objects
        """
        if not self.video_frames:
            logger.warning("No video frames available")
            return
            
        # Get start timestamp from first video frame
        start_timestamp = self.video_frames[0]["timestamp"]
        
        # Process each segment
        for segment in segments:
            # Calculate time range in the video
            start_time_rel = segment.start_time
            end_time_rel = segment.end_time
            
            # Find frames in this time range
            matching_frames = []
            for frame_data in self.video_frames:
                frame_time = frame_data["timestamp"] - start_timestamp
                if start_time_rel <= frame_time <= end_time_rel:
                    if frame_data["face_data"] is not None:
                        matching_frames.append(frame_data)
            
            # Process matching frames
            if matching_frames:
                # Calculate average face angle and spoken-to status
                angles = [f["face_data"]["angle"] for f in matching_frames if f["face_data"] is not None]
                is_facing = [f["face_data"]["is_facing_camera"] for f in matching_frames if f["face_data"] is not None]
                
                if angles:
                    avg_angle = sum(angles) / len(angles)
                    is_spoken_to = sum(is_facing) / len(is_facing) > 0.5
                    
                    # Update segment
                    segment.face_angle = avg_angle
                    segment.is_spoken_to = is_spoken_to
                    
                    # Update speaker context
                    self.context_manager.update_speaker(
                        speaker_id=segment.speaker_id,
                        is_spoken_to=is_spoken_to
                    )

    def _save_results(self, segments: List[DiarizationResult]):
        """
        Save the results to a JSON file.
        
        Args:
            segments: List of DiarizationResult objects
        """
        try:
            # Convert segments to dict for JSON serialization
            segments_dict = [segment.to_dict() for segment in segments]
            
            # Create conversation data structure
            conversation_data = {
                "conversation_id": self.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "results": segments_dict
            }
            
            # Save to file
            output_file = os.path.join(self.output_dir, f"{self.conversation_id}_results.json")
            with open(output_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

def main():
    """Main function to run the processor."""
    processor = SimplifiedLaptopProcessor()
    
    try:
        # Start recording
        print("\nPress Enter to start recording...")
        input()
        processor.start_recording()
        
        # Stop on enter key
        print("Recording... Press Enter to stop.")
        input()
        processor.stop_recording()
        
        print(f"Processing complete. Results saved in '{processor.output_dir}' directory.")
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping recording...")
        processor.stop_recording()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 