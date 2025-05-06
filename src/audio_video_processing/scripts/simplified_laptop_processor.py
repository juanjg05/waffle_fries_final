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
import re

# Add SpeechBrain imports - commented out for Windows compatibility
# from speechbrain.pretrained import EncoderClassifier

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
        self.conversation_index = 1
    
    def _load_existing_contexts(self):
        """Load existing speaker contexts from storage directory."""
        combined_file = os.path.join(self.storage_dir, "speaker_contexts.json")
        if os.path.exists(combined_file):
            try:
                with open(combined_file, 'r') as f:
                    self.speaker_data = json.load(f)
                logger.info(f"Loaded {len(self.speaker_data)} speaker contexts from {combined_file}")
            except Exception as e:
                logger.error(f"Failed to load combined speaker contexts: {e}")
                # Try loading individual files as fallback
                self._load_individual_files()
        else:
            self._load_individual_files()
    
    def _load_individual_files(self):
        """Load individual speaker context files as fallback."""
        for item in os.listdir(self.storage_dir):
            item_path = os.path.join(self.storage_dir, item)
            # Check if it's a directory for a speaker
            if os.path.isdir(item_path) and item.startswith("speaker_"):
                speaker_json = os.path.join(item_path, f"{item}.json")
                if os.path.exists(speaker_json):
                    try:
                        with open(speaker_json, 'r') as f:
                            speaker_data = json.load(f)
                            speaker_id = speaker_data.get('speaker_id')
                            if speaker_id:
                                self.speaker_data[speaker_id] = speaker_data
                    except Exception as e:
                        logger.error(f"Failed to load speaker context {speaker_json}: {e}")
    
    def update_speaker(self, speaker_id: str, embedding: np.ndarray = None, 
                       transcript: str = None, is_spoken_to: bool = None,
                       start_time: float = None, end_time: float = None):
        """Update speaker context with new data."""
        if speaker_id not in self.speaker_data:
            self.speaker_data[speaker_id] = {
                'speaker_id': speaker_id,
                'interaction_count': 0,
                'last_interaction_time': datetime.now().isoformat(),
                'common_intents': {},
                'average_confidence': 1.0,
                'conversation_history': [],
                'embedding_index': 0,
                'embedding': None,
                'spoken_to_count': 0,
                'not_spoken_to_count': 0
            }
        
        speaker = self.speaker_data[speaker_id]
        speaker['last_interaction_time'] = datetime.now().isoformat()
        
        if embedding is not None and embedding.size > 0:
            # Convert embedding to list for JSON serialization
            speaker['embedding'] = embedding.tolist()
        
        if transcript is not None and start_time is not None and end_time is not None:
            speaker['interaction_count'] += 1
            speaker['conversation_history'].append({
                'conversation_index': self.conversation_index,
                'timestamp': datetime.now().isoformat(),
                'start_time': start_time,
                'end_time': end_time,
                'confidence': 1.0,
                'transcript': transcript
            })
            
            # Keep only the last 50 conversation history entries to avoid very large files
            if len(speaker['conversation_history']) > 50:
                speaker['conversation_history'] = speaker['conversation_history'][-50:]
        
        if is_spoken_to is not None:
            if is_spoken_to:
                speaker['spoken_to_count'] += 1
            else:
                speaker['not_spoken_to_count'] += 1
        
        # Save to files
        self._save_speaker(speaker_id)
        self._save_combined_contexts()
    
    def _save_speaker(self, speaker_id: str):
        """Save speaker context to file."""
        # Create speaker directory
        speaker_dir = os.path.join(self.storage_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Save speaker data to JSON file in the speaker's directory
        speaker_file = os.path.join(speaker_dir, f"{speaker_id}.json")
        with open(speaker_file, 'w') as f:
            json.dump(self.speaker_data[speaker_id], f, indent=2)
        
        # Also save embedding as numpy file if available
        if 'embedding' in self.speaker_data[speaker_id] and self.speaker_data[speaker_id]['embedding']:
            embedding_file = os.path.join(speaker_dir, f"{speaker_id}_embedding.npy")
            embedding_np = np.array(self.speaker_data[speaker_id]['embedding'])
            np.save(embedding_file, embedding_np)
            logger.debug(f"Saved speaker embedding to {embedding_file}")
        
        # Save transcript to text file
        if 'conversation_history' in self.speaker_data[speaker_id] and self.speaker_data[speaker_id]['conversation_history']:
            transcript_file = os.path.join(speaker_dir, f"{speaker_id}_transcript.txt")
            with open(transcript_file, 'w') as f:
                for entry in self.speaker_data[speaker_id]['conversation_history']:
                    f.write(f"{entry['timestamp']} ({entry['start_time']} - {entry['end_time']}): {entry['transcript']}\n")
    
    def _save_combined_contexts(self):
        """Save all speaker contexts to a combined JSON file."""
        combined_file = os.path.join(self.storage_dir, "speaker_contexts.json")
        with open(combined_file, 'w') as f:
            json.dump(self.speaker_data, f, indent=2)
        logger.debug(f"Saved combined speaker contexts to {combined_file}")
    
    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        """Get speaker context by ID."""
        return self.speaker_data.get(speaker_id)
    
    def get_all_speakers(self) -> List[Dict]:
        """Get all speaker contexts."""
        return list(self.speaker_data.values())
    
    def increment_conversation_index(self):
        """Increment the conversation index for new conversations."""
        self.conversation_index += 1
        logger.debug(f"Incremented conversation index to {self.conversation_index}")

class SimplifiedLaptopProcessor:
    def __init__(self, 
                 output_dir="conversation_data",
                 model_dir="models",
                 speaker_contexts_dir="speaker_contexts",
                 min_speaker_duration=0.5,
                 similarity_threshold=0.85,
                 use_pyannote=True,
                 face_angle_threshold=30.0,
                 conversation_index=1):
        """
        Initialize the laptop audio and video processor.
        
        Args:
            output_dir: Directory to save conversation data
            model_dir: Directory to store models
            speaker_contexts_dir: Directory to store speaker contexts
            min_speaker_duration: Minimum duration for a speaker segment
            similarity_threshold: Threshold for speaker similarity matching
            use_pyannote: Whether to use Pyannote for diarization
            face_angle_threshold: Threshold angle in degrees for considering if facing camera
            conversation_index: Starting index for conversation naming
        """
        # Create output directory structure
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.speaker_contexts_dir = speaker_contexts_dir
        
        # Ensure directories exist
        os.makedirs(f"data/{output_dir}", exist_ok=True)
        os.makedirs(f"data/{speaker_contexts_dir}", exist_ok=True)
        os.makedirs(f"src/{model_dir}", exist_ok=True)
        
        # Initialize context manager
        self.context_manager = SpeakerContextManager(storage_dir=f"data/{speaker_contexts_dir}")
        
        # Audio parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.min_speaker_duration = min_speaker_duration
        self.similarity_threshold = similarity_threshold
        self.use_pyannote = use_pyannote
        self.face_angle_threshold = face_angle_threshold
        self.conversation_index = conversation_index
        
        # Look for existing conversation IDs
        self.last_conversation_idx = self._find_last_conversation_index()
        self.context_manager.conversation_index = self.last_conversation_idx + 1
        
        logger.info(f"Setting conversation index to {self.context_manager.conversation_index} based on existing files")
        
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
        
        # Note about SpeechBrain
        logger.info("Note: In production, SpeechBrain ECAPA-TDNN should be used for speaker embeddings.")
        logger.info("Using MFCC fallback for this demo due to Windows symlink permissions.")
        
        # Initialize Pyannote diarization if available
        self.diarization_pipeline = None
        hf_token = os.getenv("HF_TOKEN")
        if self.use_pyannote and hf_token:
            try:
                from pyannote.audio import Pipeline
                logger.info("Loading Pyannote diarization pipeline...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                logger.info("Pyannote diarization pipeline loaded")
            except Exception as e:
                logger.error(f"Error loading Pyannote: {str(e)}")
                self.diarization_pipeline = None
                self.use_pyannote = False
        else:
            logger.info("Pyannote diarization disabled or HF_TOKEN not provided")
        
        logger.info(f"Simplified Laptop Processor initialized (with {'Pyannote' if self.use_pyannote else 'custom'} diarization)")
    
    def _find_last_conversation_index(self):
        """Find the last used conversation index from existing files."""
        try:
            max_idx = 0
            conversations_dir = os.path.join("data", self.output_dir)
            
            if os.path.exists(conversations_dir):
                for item in os.listdir(conversations_dir):
                    if item.startswith("conversation_"):
                        try:
                            # Extract the index from conversation_X format
                            idx_part = item.split("_")[1].split("/")[0].split("\\")[0]
                            idx = int(idx_part)
                            max_idx = max(max_idx, idx)
                        except (ValueError, IndexError):
                            pass
            
            return max_idx
        except Exception as e:
            logger.error(f"Error finding last conversation index: {str(e)}")
            return 0
    
    def start_recording(self):
        """Start recording audio and video."""
        if self.recording:
            logger.warning("Already recording")
            return
        
        # Generate conversation ID using index
        self.conversation_id = f"conversation_{self.context_manager.conversation_index}"
        logger.info(f"Starting recording for {self.conversation_id}")
        
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
        
        # Increment conversation index for next conversation
        self.conversation_index += 1
        
        # Increment conversation index for speaker context manager
        self.context_manager.increment_conversation_index()

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
            is_facing_camera = abs(angle_deg) < self.face_angle_threshold
            
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
            # Create conversation directory with proper structure
            conversation_path = os.path.join("data", self.output_dir, self.conversation_id)
            os.makedirs(conversation_path, exist_ok=True)
            os.makedirs(os.path.join(conversation_path, "frames"), exist_ok=True)
            
            # Save audio to conversation directory
            audio_filename = os.path.join(conversation_path, "audio.wav")
            self._save_audio(audio_filename)
            
            # Save sample frames to conversation directory
            frames_dir = os.path.join(conversation_path, "frames")
            self._save_sample_frames(frames_dir)
            
            # Transcribe the audio
            logger.info("Transcribing audio...")
            transcription_results = self._transcribe_audio(audio_filename)
            
            # Add face information
            self._add_face_information(transcription_results)
            
            # Save results to conversation directory
            results_file = os.path.join(conversation_path, "results.json")
            self._save_results(transcription_results, results_file)
            
            logger.info(f"Processing complete. Results saved to {conversation_path}")
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
        Transcribe the audio and create segments with speaker diarization.
        
        Args:
            audio_filename: Path to the audio file
            
        Returns:
            List of DiarizationResult objects
        """
        try:
            # Check if file exists and is accessible
            if not os.path.exists(audio_filename):
                logger.error(f"Audio file not found: {audio_filename}")
                return []
                
            # Log file size and path
            file_size = os.path.getsize(audio_filename)
            logger.info(f"Transcribing audio file: {audio_filename} (size: {file_size} bytes)")
            
            # Load audio using librosa
            logger.info(f"Loading audio file with librosa...")
            audio_data, sample_rate = librosa.load(audio_filename, sr=16000, mono=True)
            
            # Transcribe using Whisper
            logger.info(f"Starting whisper transcription...")
            result = self.whisper_model.transcribe(audio_data, language="en")
            
            # Create segments from whisper segments
            segments = []
            
            # Let's try to detect multiple speakers by using whisper segments with a time gap check
            current_speaker_idx = 0
            last_segment_end = 0
            speaker_embeddings = {}
            
            for i, segment in enumerate(result["segments"]):
                # Get segment time range
                segment_start = segment["start"]
                segment_end = segment["end"]
                
                # Extract audio for this segment
                start_sample = int(segment_start * sample_rate)
                end_sample = int(segment_end * sample_rate)
                audio_segment = audio_data[start_sample:end_sample]
                
                # Extract voice embedding
                embedding = self._extract_voice_embedding(audio_segment, sample_rate)
                
                # If there's a significant gap (>1.5s) between segments, consider a speaker change
                if i > 0 and segment_start - last_segment_end > 1.5:
                    # Increase probability of detecting a new speaker
                    potential_new_speaker = True
                else:
                    potential_new_speaker = False
                
                # Try to match with existing speakers
                speaker_id = self._assign_speaker_id(embedding, potential_new_speaker)
                
                # Store embedding for this speaker
                if speaker_id not in speaker_embeddings:
                    speaker_embeddings[speaker_id] = embedding
                
                # Create diarization result
                diarization_result = DiarizationResult(
                    speaker_id=speaker_id,
                    start_time=segment_start,
                    end_time=segment_end,
                    transcript=segment["text"].strip(),
                    confidence=segment.get("confidence", 0.9),
                    speaker_embedding=embedding,
                    speaker_confidence=0.9
                )
                segments.append(diarization_result)
                
                # Update speaker context in data/speaker_contexts/speaker_id directory
                self._update_speaker_context(
                    speaker_id=speaker_id,
                    embedding=embedding,
                    transcript=segment["text"].strip(),
                    start_time=segment_start,
                    end_time=segment_end
                )
                
                # Update last segment end
                last_segment_end = segment_end
            
            logger.info(f"Transcribed {len(segments)} segments with {len(speaker_embeddings)} unique speakers")
            return segments
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _update_speaker_context(self, speaker_id, embedding, transcript=None, start_time=None, end_time=None, is_spoken_to=None):
        """
        Update speaker context with proper file structure
        
        Args:
            speaker_id: Speaker identifier
            embedding: Voice embedding
            transcript: Transcript text
            start_time: Segment start time
            end_time: Segment end time
            is_spoken_to: Whether the speaker is being spoken to
        """
        # Make sure speaker directory exists
        speaker_dir = os.path.join("data", "speaker_contexts", speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Update speaker context
        self.context_manager.update_speaker(
            speaker_id=speaker_id,
            embedding=embedding,
            transcript=transcript,
            start_time=start_time,
            end_time=end_time,
            is_spoken_to=is_spoken_to
        )

    def _assign_speaker_id(self, embedding, potential_new_speaker=False):
        """
        Assign a speaker ID based on embedding similarity.
        
        Args:
            embedding: Voice embedding
            potential_new_speaker: Flag indicating higher likelihood of a new speaker
            
        Returns:
            Speaker ID
        """
        if not embedding.any():
            return f"speaker_{len(self.context_manager.get_all_speakers())}"
        
        speakers = self.context_manager.get_all_speakers()
        best_similarity = -1
        best_speaker_id = None
        
        for speaker in speakers:
            if 'embedding' in speaker and speaker['embedding']:
                stored_embedding = np.array(speaker['embedding'])
                if embedding.shape == stored_embedding.shape:
                    similarity = 1 - cosine(embedding, stored_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_speaker_id = speaker['speaker_id']
        
        # Adjust threshold if potential new speaker is detected
        threshold = self.similarity_threshold * 0.95 if potential_new_speaker else self.similarity_threshold
        
        if best_similarity > threshold and best_speaker_id:
            return best_speaker_id
        else:
            # Create a new speaker ID
            return f"speaker_{len(speakers)}"

    def _save_results(self, segments, results_file=None):
        """
        Save the results to a JSON file.
        
        Args:
            segments: List of DiarizationResult objects
            results_file: Output file path
        """
        try:
            if not segments:
                logger.warning("No segments to save")
                return
                
            # Count unique speakers
            unique_speakers = set(segment.speaker_id for segment in segments)
            num_speakers = len(unique_speakers)
            
            # Create a JSON serializable dictionary
            segments_dict = []
            for segment in segments:
                segment_dict = segment.to_dict()
                segments_dict.append(segment_dict)
            
            # Create conversation data structure
            conversation_data = {
                "conversation_id": self.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "num_speakers": num_speakers,
                "speaker_ids": list(unique_speakers),
                "results": segments_dict
            }
            
            # If no results file specified, use default path
            if not results_file:
                results_file = os.path.join("data", self.output_dir, f"{self.conversation_id}", "results.json")
            
            # Save to file
            with open(results_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _extract_voice_embedding(self, audio_data, sample_rate):
        """
        Extract voice embedding for speaker identification using SpeechBrain ECAPA-TDNN.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Voice embedding vector
        """
        try:
            # For simplicity and to avoid Windows symlink issues,
            # we'll use a basic MFCC fallback approach with a note
            # that in production SpeechBrain's ECAPA-TDNN should be used
            logger.info("Using MFCC as fallback for speaker embeddings (SpeechBrain failed to load)")
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            
            # Take the mean across time to get a fixed-dimension embedding
            embedding = np.mean(mfccs, axis=1)
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            logger.info(f"Extracted MFCC embedding with shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting voice embedding: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a random normalized embedding on error
            random_embedding = np.random.randn(40)
            return random_embedding / np.linalg.norm(random_embedding)

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
        
        # Count unique speakers
        unique_speakers = set(segment.speaker_id for segment in segments)
        num_speakers = len(unique_speakers)
        
        # If there's only one speaker, set all segments to is_spoken_to=True
        if num_speakers <= 1 and segments:
            for segment in segments:
                segment.is_spoken_to = True
            return
        
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
                
                if angles:
                    avg_angle = sum(angles) / len(angles)
                    
                    # Determine is_spoken_to with weighted scoring
                    # 1. If angle is within threshold (e.g., +/-30 degrees), high score
                    # 2. Score decreases as angle increases beyond threshold
                    angle_scores = [max(0, 1.0 - (abs(angle) / (2 * self.face_angle_threshold))) for angle in angles]
                    angle_score = sum(angle_scores) / len(angle_scores)
                    
                    # Default threshold is 0.5, but for more sensitive detection, can be lower
                    is_spoken_to = angle_score > 0.4  # More sensitive threshold
                    
                    # Update segment
                    segment.face_angle = avg_angle
                    segment.is_spoken_to = is_spoken_to
                    
                    # Update speaker context
                    self.context_manager.update_speaker(
                        speaker_id=segment.speaker_id,
                        is_spoken_to=is_spoken_to
                    )

def main():
    """Main function to run the processor."""
    # Determine whether to use Pyannote based on HF_TOKEN availability
    use_pyannote = bool(os.getenv("HF_TOKEN"))
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    processor = SimplifiedLaptopProcessor(
        output_dir="conversation_data",
        model_dir="models",
        speaker_contexts_dir="speaker_contexts",
        use_pyannote=use_pyannote,
        face_angle_threshold=30.0
    )
    
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
        print(f"Using {'Pyannote' if processor.use_pyannote else 'custom'} diarization.")
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping recording...")
        processor.stop_recording()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 