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
from scipy.spatial.distance import cosine
import soundfile as sf
import librosa
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from dataclasses import dataclass

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
        is_spoken_to_value = bool(self.is_spoken_to)
        return {
            "speaker_id": self.speaker_id,
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "transcript": self.transcript,
            "diarization_confidence": float(self.confidence),
            "speaker_confidence": float(self.speaker_confidence),
            "speaker_embedding": self.speaker_embedding.tolist() if self.speaker_embedding is not None else None,
            "is_spoken_to": is_spoken_to_value,
            "face_angle": float(self.face_angle) if self.face_angle is not None else None
        }

class SpeakerContextManager:
    def __init__(self, storage_dir="speaker_contexts"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.speaker_data = {}
        self._load_existing_contexts()
    
    def _load_existing_contexts(self):
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.storage_dir, filename), 'r') as f:
                        speaker_data = json.load(f)
                        self.speaker_data[speaker_data.get('speaker_id')] = speaker_data
                except Exception as e:
                    print(f"Failed to load speaker context {filename}: {e}")
    
    def update_speaker(self, speaker_id: str, embedding: np.ndarray = None, 
                      transcript: str = None, is_spoken_to: bool = None):
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
        
        self._save_speaker(speaker_id)
    
    def _save_speaker(self, speaker_id: str):
        with open(os.path.join(self.storage_dir, f"{speaker_id}.json"), 'w') as f:
            json.dump(self.speaker_data[speaker_id], f, indent=2)
    
    def get_speaker(self, speaker_id: str) -> Optional[Dict]:
        return self.speaker_data.get(speaker_id)
    
    def get_all_speakers(self) -> List[Dict]:
        return list(self.speaker_data.values())

class LaptopAudioVideoProcessor:
    def __init__(self, 
                 output_dir="conversation_data",
                 model_dir="models",
                 min_speaker_duration=0.5,
                 similarity_threshold=0.85,
                 use_huggingface_token=True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.context_manager = SpeakerContextManager()
        
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.min_speaker_duration = min_speaker_duration
        self.similarity_threshold = similarity_threshold
        
        self.pyaudio = pyaudio.PyAudio()
        self.audio_frames = []
        self.recording = False
        self.audio_thread = None
        
        self.cap = None
        self.video_frames = []
        self.recording_video = False
        self.video_thread = None
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.conversation_id = None
        self.conversation_data = []
        self.current_speakers = {}
        
        try:
            if use_huggingface_token:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    print("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
            
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=hf_token if use_huggingface_token else None
                )
            except Exception as e:
                print("Will run without speaker diarization")
                self.diarization_pipeline = None
            
            self.asr_model = whisper.load_model("base")
            
            try:
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            except Exception as e:
                print("Will run without speaker embeddings")
                self.embedding_model = None
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise
    
    def start_recording(self):
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting new conversation: {self.conversation_id}")
        
        self.recording = True
        self.audio_frames = []
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        self.recording_video = True
        self.video_frames = []
        self.cap = cv2.VideoCapture(0)  
        self.video_thread = threading.Thread(target=self._record_video)
        self.video_thread.start()
        
        print("Recording started")
    
    def stop_recording(self):
        print("Stopping recording")
        
        self.recording = False
        if self.audio_thread:
            self.audio_thread.join()
        
        self.recording_video = False
        if self.video_thread:
            self.video_thread.join()
        
        if self.cap:
            self.cap.release()
        
        if not self.audio_frames:
            print("No audio data recorded")
            return None
        
        # Create conversation directory
        conversation_dir = os.path.join(self.output_dir, self.conversation_id)
        os.makedirs(conversation_dir, exist_ok=True)
        
        # Save audio file
        audio_filename = os.path.join(conversation_dir, "audio.wav")
        self._save_audio(audio_filename)
        
        # Save sample video frames
        self._save_sample_frames()
        
        # Process the conversation
        results = self._process_conversation()
        
        return results
    
    def _record_audio(self):
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
            print(f"Error recording audio: {e}")
    
    def _record_video(self):
        try:
            while self.recording_video and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                face_info = self._detect_face_direction(frame)
                self.video_frames.append((frame, face_info, time.time()))
                
                if len(self.video_frames) % 30 == 0:  # Reduce processing load
                    self._display_frame_with_info(frame.copy(), face_info)
                
                time.sleep(0.03)  # ~30 fps
        
        except Exception as e:
            print(f"Error recording video: {e}")
    
    def _detect_face_direction(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {
                    'detected': False,
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
            
            # Calculate angle from nose direction
            dx = nose_bridge[1].x - nose_bridge[0].x
            dy = nose_bridge[1].y - nose_bridge[0].y
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Determine if looking towards camera
            if abs(angle) < 30:  # Threshold for "towards"
                direction = 'towards'
                is_spoken_to = True
            else:
                direction = 'away'
                is_spoken_to = False
            
            return {
                'detected': True,
                'direction': direction,
                'angle': float(angle),
                'confidence': 0.9,
                'is_spoken_to': is_spoken_to
            }
            
        except Exception as e:
            print(f"Error detecting face direction: {e}")
            return {
                'detected': False,
                'direction': 'unknown',
                'angle': None,
                'confidence': 0.0,
                'is_spoken_to': False
            }
    
    def _display_frame_with_info(self, frame, face_info):
        if not face_info['detected']:
            cv2.putText(frame, "No face detected", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            direction = face_info['direction']
            angle = face_info['angle']
            is_spoken_to = face_info['is_spoken_to']
            
            color = (0, 255, 0) if direction == 'towards' else (0, 0, 255)
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if angle is not None:
                cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            status = "Spoken To" if is_spoken_to else "Not Spoken To"
            cv2.putText(frame, status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Face Detection', frame)
        cv2.waitKey(1)
    
    def _process_conversation(self):
        if not self.audio_frames:
            return []
        
        conversation_dir = os.path.join(self.output_dir, self.conversation_id)
        audio_filename = os.path.join(conversation_dir, "audio.wav")
        
        # Transcribe and diarize audio
        diarization_results = self._transcribe_and_diarize(audio_filename)
        
        # Save results to file
        results_file = os.path.join(conversation_dir, "conversation_data.json")
        
        # Convert results to serializable format
        serialized_results = [result.to_dict() for result in diarization_results]
        
        with open(results_file, 'w') as f:
            json.dump({
                'conversation_id': self.conversation_id,
                'timestamp': datetime.now().isoformat(),
                'duration': diarization_results[-1].end_time if diarization_results else 0,
                'speaker_count': len({r.speaker_id for r in diarization_results}),
                'results': serialized_results,
                'face_info': self._get_average_face_info()
            }, f, indent=2)
        
        print(f"Conversation data saved to {results_file}")
        return diarization_results
    
    def _save_audio(self, filename):
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))
            
            print(f"Audio saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
    
    def _save_sample_frames(self):
        if not self.video_frames:
            return
        
        conversation_dir = os.path.join(self.output_dir, self.conversation_id)
        frames_dir = os.path.join(conversation_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save every 30th frame
        for i, (frame, face_info, timestamp) in enumerate(self.video_frames[::30]):
            frame_filename = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        print(f"Saved {len(self.video_frames)//30} sample frames")
    
    def _transcribe_and_diarize(self, audio_filename) -> List[DiarizationResult]:
        results = []
        
        try:
            # Transcribe with Whisper
            print("Transcribing audio...")
            transcription = self.asr_model.transcribe(
                audio_filename,
                word_timestamps=True
            )
            
            full_transcript = transcription.get("text", "")
            print(f"Transcript: {full_transcript}")
            
            # Extract word timestamps
            words_with_timestamps = []
            for segment in transcription.get("segments", []):
                for word in segment.get("words", []):
                    words_with_timestamps.append({
                        'word': word.get('word', ''),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0),
                        'confidence': word.get('confidence', 0)
                    })
            
            # If diarization is available
            if self.diarization_pipeline:
                print("Performing speaker diarization...")
                diarization = self.diarization_pipeline(audio_filename)
                
                # Extract audio data for speaker embeddings
                audio_data, _ = sf.read(audio_filename)
                
                # Process each speaker segment
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_time = turn.start
                    end_time = turn.end
                    duration = end_time - start_time
                    
                    # Skip very short segments
                    if duration < self.min_speaker_duration:
                        continue
                    
                    # Find words in this segment
                    segment_words = []
                    for word_info in words_with_timestamps:
                        word_start = word_info['start']
                        word_end = word_info['end']
                        
                        # Check if word is fully contained in segment
                        if word_start >= start_time and word_end <= end_time:
                            segment_words.append(word_info['word'])
                    
                    # Create segment transcript
                    segment_transcript = " ".join(segment_words)
                    
                    # Get speaker embedding
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = min(int(end_time * self.sample_rate), len(audio_data))
                    speaker_audio = audio_data[start_sample:end_sample]
                    speaker_embedding = self._get_speaker_embedding(speaker_audio)
                    
                    # Get face info for this time period
                    face_info = self._match_face_with_audio_time(start_time, end_time)
                    
                    # Create result
                    result = DiarizationResult(
                        speaker_id=speaker,
                        start_time=start_time,
                        end_time=end_time,
                        transcript=segment_transcript,
                        confidence=1.0,
                        speaker_embedding=speaker_embedding,
                        speaker_confidence=1.0 if speaker_embedding is not None else 0.0,
                        is_spoken_to=face_info.get('is_spoken_to', False),
                        face_angle=face_info.get('angle')
                    )
                    
                    # Update speaker context
                    self.context_manager.update_speaker(
                        speaker_id=speaker,
                        embedding=speaker_embedding,
                        transcript=segment_transcript,
                        is_spoken_to=face_info.get('is_spoken_to', False)
                    )
                    
                    results.append(result)
            else:
                # Without diarization, create one segment for the entire audio
                print("Diarization not available. Creating single speaker segment.")
                result = DiarizationResult(
                    speaker_id="unknown_speaker",
                    start_time=0.0,
                    end_time=transcription.get("segments", [{}])[-1].get("end", 0.0),
                    transcript=full_transcript,
                    confidence=1.0,
                    speaker_embedding=None,
                    speaker_confidence=0.0,
                    is_spoken_to=True  # Assume spoken to without face info
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            print(f"Error in transcription and diarization: {e}")
            return results
    
    def _get_speaker_embedding(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        if self.embedding_model is None or len(audio_segment) < 1000:
            return None
        
        try:
            # Convert to float tensor
            waveform = torch.FloatTensor(audio_segment).unsqueeze(0)
            
            # Extract embedding
            return self.embedding_model(waveform).detach().cpu().numpy()
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            return None
    
    def _match_face_with_audio_time(self, start_time, end_time):
        if not self.video_frames:
            return {'detected': False, 'is_spoken_to': False}
        
        # Calculate approximate recording start time
        recording_start = self.video_frames[0][2]  # First frame timestamp
        
        # Calculate absolute start and end times
        abs_start = recording_start + start_time
        abs_end = recording_start + end_time
        
        # Find frames in the time range
        matching_frames = []
        for frame, face_info, timestamp in self.video_frames:
            if abs_start <= timestamp <= abs_end:
                matching_frames.append((frame, face_info, timestamp))
        
        if not matching_frames:
            return {'detected': False, 'is_spoken_to': False}
        
        # Count directions
        directions = {"towards": 0, "away": 0, "unknown": 0}
        angles = []
        is_spoken_to_count = 0
        detected_count = 0
        
        for _, face_info, _ in matching_frames:
            if face_info.get('detected', False):
                detected_count += 1
                directions[face_info.get('direction', 'unknown')] += 1
                
                if face_info.get('angle') is not None:
                    angles.append(face_info.get('angle'))
                
                if face_info.get('is_spoken_to', False):
                    is_spoken_to_count += 1
        
        if detected_count == 0:
            return {'detected': False, 'is_spoken_to': False}
        
        # Calculate the most common direction and average angle
        max_direction = max(directions.items(), key=lambda x: x[1])[0]
        avg_angle = sum(angles) / len(angles) if angles else None
        is_spoken_to = is_spoken_to_count > (detected_count / 2)  # Majority vote
        
        return {
            'detected': True,
            'direction': max_direction,
            'angle': avg_angle,
            'is_spoken_to': is_spoken_to,
            'confidence': detected_count / len(matching_frames)
        }
    
    def _get_average_face_info(self):
        if not self.video_frames:
            return {
                'detected_ratio': 0,
                'towards_ratio': 0,
                'away_ratio': 0,
                'unknown_ratio': 0,
                'spoken_to_ratio': 0,
                'average_angle': None
            }
        
        directions = {"towards": 0, "away": 0, "unknown": 0}
        angles = []
        detected_count = 0
        spoken_to_count = 0
        
        for _, face_info, _ in self.video_frames:
            if face_info.get('detected', False):
                detected_count += 1
                directions[face_info.get('direction', 'unknown')] += 1
                
                if face_info.get('angle') is not None:
                    angles.append(face_info.get('angle'))
                
                if face_info.get('is_spoken_to', False):
                    spoken_to_count += 1
        
        total_frames = len(self.video_frames)
        
        return {
            'detected_ratio': detected_count / total_frames if total_frames > 0 else 0,
            'towards_ratio': directions['towards'] / total_frames if total_frames > 0 else 0,
            'away_ratio': directions['away'] / total_frames if total_frames > 0 else 0,
            'unknown_ratio': directions['unknown'] / total_frames if total_frames > 0 else 0,
            'spoken_to_ratio': spoken_to_count / total_frames if total_frames > 0 else 0,
            'average_angle': sum(angles) / len(angles) if angles else None
        }

def main():
    import argparse
    import wave
    
    parser = argparse.ArgumentParser(description="Laptop Audio/Video Processor")
    parser.add_argument("--output-dir", type=str, default="conversation_data",
                       help="Directory to save conversation data")
    parser.add_argument("--audio-file", type=str,
                       help="Process existing audio file instead of recording new audio")
    
    args = parser.parse_args()
    
    processor = LaptopAudioVideoProcessor(output_dir=args.output_dir)
    
    if args.audio_file:
        if os.path.exists(args.audio_file):
            print(f"Processing audio file: {args.audio_file}")
            processor.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_dir = os.path.join(args.output_dir, processor.conversation_id)
            os.makedirs(conversation_dir, exist_ok=True)
            
            audio_filename = os.path.join(conversation_dir, "audio.wav")
            
            # Copy the file
            with wave.open(args.audio_file, 'rb') as src:
                with wave.open(audio_filename, 'wb') as dst:
                    dst.setnchannels(src.getnchannels())
                    dst.setsampwidth(src.getsampwidth())
                    dst.setframerate(src.getframerate())
                    dst.writeframes(src.readframes(src.getnframes()))
            
            results = processor._transcribe_and_diarize(audio_filename)
            
            # Save results
            results_file = os.path.join(conversation_dir, "conversation_data.json")
            serialized_results = [result.to_dict() for result in results]
            
            with open(results_file, 'w') as f:
                json.dump({
                    'conversation_id': processor.conversation_id,
                    'timestamp': datetime.now().isoformat(),
                    'duration': results[-1].end_time if results else 0,
                    'speaker_count': len({r.speaker_id for r in results}),
                    'results': serialized_results
                }, f, indent=2)
            
            print(f"Results saved to: {results_file}")
        else:
            print(f"Error: Audio file not found: {args.audio_file}")
    else:
        print("Recording new conversation. Press Ctrl+C to stop...")
        try:
            processor.start_recording()
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            results = processor.stop_recording()
            print("Recording stopped.")
            if results:
                print(f"Detected {len({r.speaker_id for r in results})} speakers")
                print(f"Transcription length: {sum(len(r.transcript) for r in results)} characters")

if __name__ == "__main__":
    main() 