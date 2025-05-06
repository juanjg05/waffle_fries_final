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
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
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

class LaptopAudioVideoProcessor:
    def __init__(self, 
                 output_dir="conversation_data",
                 model_dir="models",
                 min_speaker_duration=0.5,
                 similarity_threshold=0.85,
                 use_huggingface_token=True):
        """
        Initialize the laptop audio and video processor.
        
        Args:
            output_dir: Directory to save conversation data
            model_dir: Directory to store models
            min_speaker_duration: Minimum duration for a speaker segment
            similarity_threshold: Threshold for speaker similarity matching
            use_huggingface_token: Whether to use a Hugging Face token from environment
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
        
        # Load models
        try:
            # Use Hugging Face token if specified
            if use_huggingface_token:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    logger.warning(
                        "Hugging Face token not found. Please set the HF_TOKEN environment variable. "
                        "You can get a token from https://hf.co/settings/tokens"
                    )
            
            # Load PyAnnote diarization pipeline
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=hf_token if use_huggingface_token else None
                )
                logger.info("Successfully loaded PyAnnote diarization model")
            except Exception as e:
                logger.error(f"Error loading PyAnnote model: {str(e)}")
                logger.info("Will run without speaker diarization")
                self.diarization_pipeline = None
            
            # Load Whisper ASR model
            logger.info("Loading Whisper ASR model...")
            self.asr_model = whisper.load_model("base")
            logger.info("Successfully loaded Whisper model")
            
            # Load speaker embedding model
            try:
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                logger.info("Successfully loaded speaker embedding model")
            except Exception as e:
                logger.error(f"Error loading speaker embedding model: {str(e)}")
                logger.info("Will run without speaker embeddings")
                self.embedding_model = None
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def start_recording(self):
        """Start recording audio and video."""
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting new conversation: {self.conversation_id}")
        
        # Start audio recording
        self.recording = True
        self.audio_frames = []
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        # Start video recording
        self.recording_video = True
        self.video_frames = []
        self.cap = cv2.VideoCapture(0)  # Use default webcam
        self.video_thread = threading.Thread(target=self._record_video)
        self.video_thread.start()
        
        logger.info("Recording started")
    
    def stop_recording(self):
        """Stop recording and process the conversation."""
        logger.info("Stopping recording")
        
        # Stop audio recording
        self.recording = False
        if self.audio_thread:
            self.audio_thread.join()
        
        # Stop video recording
        self.recording_video = False
        if self.video_thread:
            self.video_thread.join()
        
        if self.cap:
            self.cap.release()
        
        # Process the recorded data
        self._process_conversation()
        
        logger.info("Recording stopped and processing completed")
    
    def _record_audio(self):
        """Record audio from microphone."""
        stream = self.pyaudio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        logger.info("Audio recording started")
        
        while self.recording:
            data = stream.read(self.chunk_size)
            self.audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        logger.info("Audio recording stopped")
    
    def _record_video(self):
        """Record video from webcam."""
        logger.info("Video recording started")
        
        while self.recording_video:
            ret, frame = self.cap.read()
            if ret:
                # Process face direction
                face_info = self._detect_face_direction(frame)
                
                # Add timestamp and face info to frame metadata
                timestamp = time.time()
                self.video_frames.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'face_info': face_info
                })
            
            time.sleep(0.033)  # ~30 fps
        
        logger.info("Video recording stopped")
    
    def _detect_face_direction(self, frame):
        """
        Detect face direction using MediaPipe Face Mesh.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                'direction': 'unknown',
                'angle': None,
                'confidence': 0.0
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
    
    def _process_conversation(self):
        """Process the recorded conversation."""
        # Save audio to file
        audio_filename = os.path.join(self.output_dir, f"{self.conversation_id}_audio.wav")
        self._save_audio(audio_filename)
        
        # Save a few frames for reference
        self._save_sample_frames()
        
        # Transcribe and diarize audio
        logger.info("Transcribing and diarizing audio...")
        results = self._transcribe_and_diarize(audio_filename)
        
        # Save results to file
        results_filename = os.path.join(self.output_dir, f"{self.conversation_id}_results.json")
        with open(results_filename, 'w') as f:
            json.dump({
                'conversation_id': self.conversation_id,
                'timestamp': datetime.now().isoformat(),
                'results': [r.to_dict() for r in results]
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_filename}")
        
        # Log summary
        logger.info(f"Processed conversation with {len(results)} segments")
        for result in results:
            logger.info(f"Speaker {result.speaker_id}: {result.transcript}")
    
    def _save_audio(self, filename):
        """Save recorded audio to file."""
        wf = sf.SoundFile(
            filename,
            mode='w',
            samplerate=self.sample_rate,
            channels=self.channels
        )
        
        # Convert audio frames to numpy array
        audio_data = np.frombuffer(b''.join(self.audio_frames), dtype=np.int16)
        
        # Write to file
        wf.write(audio_data.astype(np.float32) / 32768.0)  # Convert to float32 [-1.0, 1.0]
        wf.close()
        
        logger.info(f"Audio saved to {filename}")
    
    def _save_sample_frames(self):
        """Save a few sample frames from the video."""
        if not self.video_frames:
            return
        
        # Create frames directory
        frames_dir = os.path.join(self.output_dir, f"{self.conversation_id}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save a frame every second (roughly 30 frames)
        frame_count = len(self.video_frames)
        step = max(1, frame_count // 30)
        
        for i in range(0, frame_count, step):
            frame_data = self.video_frames[i]
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            # Save frame
            frame_filename = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
    
    def _transcribe_and_diarize(self, audio_filename) -> List[DiarizationResult]:
        """
        Transcribe and diarize audio file.
        
        Args:
            audio_filename: Path to audio file
            
        Returns:
            List of DiarizationResult objects
        """
        results = []
        
        try:
            # Load audio file
            audio_array, sample_rate = librosa.load(audio_filename, sr=self.sample_rate)
            
            # Transcribe with Whisper
            logger.info("Transcribing with Whisper...")
            transcription = self.asr_model.transcribe(audio_array)
            
            # Extract segments with timestamps
            segments = transcription.get('segments', [])
            
            # If we have diarization pipeline, use it
            if self.diarization_pipeline:
                logger.info("Diarizing with PyAnnote...")
                diarization = self.diarization_pipeline(audio_filename)
                
                # Convert PyAnnote diarization to our format
                for segment, track in diarization.itertracks(yield_label=True):
                    speaker_id = track
                    
                    # Find transcript for this time range
                    segment_transcript = ""
                    for whisper_segment in segments:
                        w_start = whisper_segment['start']
                        w_end = whisper_segment['end']
                        
                        # Check for overlap
                        if (w_start <= segment.end and w_end >= segment.start):
                            segment_transcript += whisper_segment['text'] + " "
                    
                    # Get speaker embedding
                    speaker_embedding = None
                    if self.embedding_model:
                        start_sample = int(segment.start * sample_rate)
                        end_sample = int(segment.end * sample_rate)
                        if end_sample > start_sample:
                            segment_audio = audio_array[start_sample:end_sample]
                            if len(segment_audio) > 0:
                                speaker_embedding = self._get_speaker_embedding(segment_audio)
                    
                    # Match with face info
                    face_info = self._match_face_with_audio_time(segment.start, segment.end)
                    
                    result = DiarizationResult(
                        speaker_id=speaker_id,
                        start_time=segment.start,
                        end_time=segment.end,
                        transcript=segment_transcript.strip(),
                        confidence=0.8,  # Placeholder confidence
                        speaker_embedding=speaker_embedding,
                        speaker_confidence=0.8,  # Placeholder confidence
                        is_spoken_to=face_info.get('is_spoken_to', False) if face_info else False,
                        face_angle=face_info.get('angle') if face_info else None
                    )
                    
                    results.append(result)
                    
                    # Update speaker context
                    self.context_manager.update_speaker(
                        speaker_id=speaker_id,
                        embedding=speaker_embedding,
                        transcript=segment_transcript.strip(),
                        is_spoken_to=result.is_spoken_to
                    )
            else:
                # If no diarization, treat everything as one speaker
                logger.info("No diarization model available, treating as single speaker")
                
                speaker_id = "speaker_0"
                full_transcript = transcription.get('text', '')
                
                # Get speaker embedding for the whole audio
                speaker_embedding = None
                if self.embedding_model and len(audio_array) > 0:
                    speaker_embedding = self._get_speaker_embedding(audio_array)
                
                # Get average face direction
                face_info = self._get_average_face_info()
                
                result = DiarizationResult(
                    speaker_id=speaker_id,
                    start_time=0,
                    end_time=len(audio_array) / sample_rate,
                    transcript=full_transcript,
                    confidence=0.8,  # Placeholder confidence
                    speaker_embedding=speaker_embedding,
                    speaker_confidence=0.8,  # Placeholder confidence
                    is_spoken_to=face_info.get('is_spoken_to', False) if face_info else False,
                    face_angle=face_info.get('angle') if face_info else None
                )
                
                results.append(result)
                
                # Update speaker context
                self.context_manager.update_speaker(
                    speaker_id=speaker_id,
                    embedding=speaker_embedding,
                    transcript=full_transcript,
                    is_spoken_to=result.is_spoken_to
                )
        
        except Exception as e:
            logger.error(f"Error in transcribe_and_diarize: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return results
    
    def _get_speaker_embedding(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """Get speaker embedding from audio segment."""
        if self.embedding_model is None:
            return None
        
        try:
            # Ensure audio is float32 in range [-1, 1]
            if audio_segment.dtype != np.float32:
                audio_segment = audio_segment.astype(np.float32)
            
            if np.max(np.abs(audio_segment)) > 1.0:
                audio_segment = audio_segment / 32768.0  # Normalize if int16
            
            # Get embedding
            embedding = self.embedding_model(audio_segment)
            return embedding.squeeze().cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error getting speaker embedding: {str(e)}")
            return None
    
    def _match_face_with_audio_time(self, start_time, end_time):
        """Match face information with audio time segment."""
        if not self.video_frames:
            return None
        
        # Adjust for potential timing differences
        recording_start_time = self.video_frames[0]['timestamp']
        
        # Filter frames within the time range
        matching_frames = []
        for frame_data in self.video_frames:
            frame_time = frame_data['timestamp'] - recording_start_time
            if start_time <= frame_time <= end_time:
                matching_frames.append(frame_data)
        
        if not matching_frames:
            return None
        
        # Aggregate face info
        directions = []
        angles = []
        is_spoken_to_count = 0
        
        for frame_data in matching_frames:
            face_info = frame_data.get('face_info')
            if face_info:
                if face_info.get('direction') != 'unknown':
                    directions.append(face_info.get('direction'))
                
                if face_info.get('angle') is not None:
                    angles.append(face_info.get('angle'))
                
                if face_info.get('is_spoken_to', False):
                    is_spoken_to_count += 1
        
        if not directions and not angles:
            return None
        
        # Calculate most common direction and average angle
        most_common_direction = max(set(directions), key=directions.count) if directions else 'unknown'
        average_angle = sum(angles) / len(angles) if angles else None
        is_spoken_to = is_spoken_to_count > len(matching_frames) / 2
        
        return {
            'direction': most_common_direction,
            'angle': average_angle,
            'is_spoken_to': is_spoken_to
        }
    
    def _get_average_face_info(self):
        """Get average face information for the entire recording."""
        if not self.video_frames:
            return None
        
        # Aggregate face info
        directions = []
        angles = []
        is_spoken_to_count = 0
        total_frames = 0
        
        for frame_data in self.video_frames:
            face_info = frame_data.get('face_info')
            if face_info:
                total_frames += 1
                
                if face_info.get('direction') != 'unknown':
                    directions.append(face_info.get('direction'))
                
                if face_info.get('angle') is not None:
                    angles.append(face_info.get('angle'))
                
                if face_info.get('is_spoken_to', False):
                    is_spoken_to_count += 1
        
        if not directions and not angles:
            return None
        
        # Calculate most common direction and average angle
        most_common_direction = max(set(directions), key=directions.count) if directions else 'unknown'
        average_angle = sum(angles) / len(angles) if angles else None
        is_spoken_to = is_spoken_to_count > total_frames / 2
        
        return {
            'direction': most_common_direction,
            'angle': average_angle,
            'is_spoken_to': is_spoken_to
        }

def main():
    processor = LaptopAudioVideoProcessor()
    
    print("Press Enter to start recording...")
    input()
    
    processor.start_recording()
    
    print("Recording... Press Enter to stop...")
    input()
    
    processor.stop_recording()
    
    print("Processing complete. Results saved to the conversation_data directory.")

if __name__ == "__main__":
    main() 