#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our processing modules
from face_detection_node import FaceDetector
from audio_processor_node import AudioProcessor
from laptop_audio_video_processor import LaptopAudioVideoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedLaptopProcessor:
    def __init__(self, 
                output_dir="integrated_results",
                audio_format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                chunk_size=1024,
                use_webcam=True):
        """
        Initialize the integrated processor for laptop use.
        
        Args:
            output_dir: Directory to save results
            audio_format: PyAudio format
            channels: Number of audio channels
            rate: Audio sample rate
            chunk_size: Audio chunk size
            use_webcam: Whether to use webcam
        """
        # Create output directories
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up audio parameters
        self.audio_format = audio_format
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        
        # Initialize recorders and processors
        self.use_webcam = use_webcam
        self.recording = False
        self.frames = []
        self.video_frames = []
        self.face_info_list = []
        
        # Initialize components
        self.pyaudio = pyaudio.PyAudio()
        self.face_detector = FaceDetector()
        self.audio_processor = AudioProcessor(output_dir=os.path.join(output_dir, "audio"))
        
        # For advanced processing
        self.advanced_processor = LaptopAudioVideoProcessor(
            output_dir=os.path.join(output_dir, "advanced")
        )
        
        logger.info("Integrated processor initialized successfully")
    
    def start_recording(self, record_seconds=None):
        """
        Start recording audio and optionally video.
        
        Args:
            record_seconds: Number of seconds to record, or None for manual stop
        """
        logger.info("Starting recording...")
        
        # Reset storage
        self.frames = []
        self.video_frames = []
        self.face_info_list = []
        
        # Set recording flag
        self.recording = True
        
        # Start audio recording thread
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        # Start video recording if enabled
        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
            self.video_thread = threading.Thread(target=self._record_video)
            self.video_thread.start()
        
        # If record_seconds is specified, sleep for that duration and then stop
        if record_seconds is not None:
            time.sleep(record_seconds)
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and process the captured data."""
        logger.info("Stopping recording...")
        
        # Set flag to stop recording
        self.recording = False
        
        # Wait for threads to finish
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        
        if self.use_webcam and hasattr(self, 'video_thread'):
            self.video_thread.join()
            self.cap.release()
        
        # Create output filenames based on timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = os.path.join(self.output_dir, f"{timestamp}_audio.wav")
        results_filename = os.path.join(self.output_dir, f"{timestamp}_results.json")
        
        # Save audio data
        self._save_audio(audio_filename)
        
        # Process audio
        audio_results = self.audio_processor.process_audio_file(audio_filename)
        
        # Save video frames if available
        video_frames_dir = None
        if self.use_webcam and self.video_frames:
            video_frames_dir = os.path.join(self.output_dir, f"{timestamp}_frames")
            os.makedirs(video_frames_dir, exist_ok=True)
            
            # Save a subset of frames (every 30th frame)
            for i, (frame, face_info, frame_time) in enumerate(self.video_frames[::30]):
                frame_filename = os.path.join(video_frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
        
        # Save combined results
        results = {
            'timestamp': timestamp,
            'audio_file': os.path.basename(audio_filename),
            'transcript': audio_results.get('transcript', ''),
            'word_count': len(audio_results.get('word_timestamps', [])),
            'duration': audio_results.get('processing_time', 0),
            'video_frames_dir': os.path.basename(video_frames_dir) if video_frames_dir else None,
            'face_info_summary': self._summarize_face_info() if self.face_info_list else None
        }
        
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_filename}")
        
        return results
    
    def process_with_advanced_features(self):
        """Use the advanced processor for more features like diarization."""
        if hasattr(self, 'advanced_processor'):
            logger.info("Starting advanced processing...")
            self.advanced_processor.start_recording()
            input("Press Enter to stop recording...")
            self.advanced_processor.stop_recording()
            logger.info("Advanced processing complete")
            return True
        return False
    
    def _record_audio(self):
        """Record audio from microphone."""
        try:
            # Open audio stream
            stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Audio recording started")
            
            # Record audio in chunks
            while self.recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            logger.info("Audio recording stopped")
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
    
    def _record_video(self):
        """Record video from webcam."""
        try:
            logger.info("Video recording started")
            
            # Record video frames with face detection
            while self.recording and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect face direction
                face_info = self.face_detector.detect_face_direction(frame)
                
                # Store frame, face info, and timestamp
                self.video_frames.append((frame, face_info, time.time()))
                
                # Add face info to list
                self.face_info_list.append(face_info)
                
                # Display preview with face direction
                self._display_frame_with_face_info(frame, face_info)
                
                # Process at approximately 30 fps
                time.sleep(0.03)
            
            logger.info("Video recording stopped")
            
        except Exception as e:
            logger.error(f"Error recording video: {e}")
    
    def _display_frame_with_face_info(self, frame, face_info):
        """Display frame with face detection information."""
        # Add face direction text to frame
        direction = face_info.get('direction', 'unknown')
        angle = face_info.get('angle')
        
        # Choose color based on direction (green for towards, red for away)
        color = (0, 255, 0) if direction == 'towards' else (0, 0, 255)
        
        # Add direction text
        cv2.putText(frame, f"Direction: {direction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add angle text if available
        if angle is not None:
            cv2.putText(frame, f"Angle: {angle:.1f} degrees", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add spoken-to status
        is_spoken_to = face_info.get('is_spoken_to', False)
        spoken_text = "Spoken to: Yes" if is_spoken_to else "Spoken to: No"
        cv2.putText(frame, spoken_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show the frame
        cv2.imshow('Face Detection', frame)
        cv2.waitKey(1)
    
    def _save_audio(self, filename):
        """Save recorded audio to WAV file."""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            logger.info(f"Audio saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
    
    def _summarize_face_info(self):
        """Summarize face detection results."""
        if not self.face_info_list:
            return None
        
        # Count directions
        directions = [info.get('direction', 'unknown') for info in self.face_info_list]
        direction_counts = {
            'towards': directions.count('towards'),
            'away': directions.count('away'),
            'unknown': directions.count('unknown')
        }
        
        # Calculate average angle when available
        angles = [info.get('angle') for info in self.face_info_list if info.get('angle') is not None]
        avg_angle = sum(angles) / len(angles) if angles else None
        
        # Count spoken-to status
        spoken_to_count = sum(1 for info in self.face_info_list if info.get('is_spoken_to', False))
        
        return {
            'total_frames': len(self.face_info_list),
            'direction_counts': direction_counts,
            'average_angle': avg_angle,
            'spoken_to_ratio': spoken_to_count / len(self.face_info_list) if self.face_info_list else 0
        }

def main():
    """Main function to run the integrated processor."""
    parser = argparse.ArgumentParser(description="Integrated Audio/Video Processor")
    
    parser.add_argument("--mode", choices=["simple", "advanced"], default="advanced",
                       help="Processing mode (simple or advanced)")
    parser.add_argument("--output-dir", default="integrated_results",
                       help="Directory to save results")
    parser.add_argument("--duration", type=int, default=None,
                       help="Recording duration in seconds (default: manual stop)")
    parser.add_argument("--no-webcam", action="store_true",
                       help="Disable webcam/video processing")
    
    args = parser.parse_args()
    
    # Create processor
    processor = IntegratedLaptopProcessor(
        output_dir=args.output_dir,
        use_webcam=not args.no_webcam
    )
    
    # Run in selected mode
    if args.mode == "advanced":
        print("=== Advanced Audio/Video Processing ===")
        print("This mode provides diarization, speaker tracking, and enhanced analysis.")
        print("Press Enter to start recording, then press Enter again to stop.")
        
        input("Press Enter to start...")
        processor.process_with_advanced_features()
        
    else:  # Simple mode
        print("=== Simple Audio/Video Recording ===")
        
        if args.duration:
            print(f"Recording for {args.duration} seconds...")
            processor.start_recording(record_seconds=args.duration)
        else:
            print("Press Enter to start recording, then press Enter again to stop.")
            input("Press Enter to start...")
            processor.start_recording()
            input("Press Enter to stop...")
            processor.stop_recording()
    
    print("\nProcessing complete! Results saved to the output directory.")

if __name__ == "__main__":
    main() 