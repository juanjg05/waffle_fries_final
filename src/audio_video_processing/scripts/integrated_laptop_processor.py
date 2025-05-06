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
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from face_detection_node import FaceDetector
from audio_processor_node import AudioProcessor
from laptop_audio_video_processor import LaptopAudioVideoProcessor

class IntegratedLaptopProcessor:
    def __init__(self, 
                output_dir="integrated_results",
                audio_format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                chunk_size=1024,
                use_webcam=True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.audio_format = audio_format
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        
        self.use_webcam = use_webcam
        self.recording = False
        self.frames = []
        self.video_frames = []
        self.face_info_list = []
        
        self.pyaudio = pyaudio.PyAudio()
        self.face_detector = FaceDetector()
        self.audio_processor = AudioProcessor(output_dir=os.path.join(output_dir, "audio"))
        
        self.advanced_processor = LaptopAudioVideoProcessor(
            output_dir=os.path.join(output_dir, "advanced")
        )
    
    def start_recording(self, record_seconds=None):
        self.frames = []
        self.video_frames = []
        self.face_info_list = []
        
        self.recording = True
        
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
            self.video_thread = threading.Thread(target=self._record_video)
            self.video_thread.start()
        
        if record_seconds is not None:
            time.sleep(record_seconds)
            self.stop_recording()
    
    def stop_recording(self):
        self.recording = False
        
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        
        if self.use_webcam and hasattr(self, 'video_thread'):
            self.video_thread.join()
            self.cap.release()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = os.path.join(self.output_dir, f"{timestamp}_audio.wav")
        results_filename = os.path.join(self.output_dir, f"{timestamp}_results.json")
        
        self._save_audio(audio_filename)
        
        audio_results = self.audio_processor.process_audio_file(audio_filename)
        
        video_frames_dir = None
        if self.use_webcam and self.video_frames:
            video_frames_dir = os.path.join(self.output_dir, f"{timestamp}_frames")
            os.makedirs(video_frames_dir, exist_ok=True)
            
            for i, (frame, face_info, frame_time) in enumerate(self.video_frames[::30]):
                frame_filename = os.path.join(video_frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
        
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
        
        print(f"Results saved to {results_filename}")
        
        return results
    
    def process_with_advanced_features(self):
        if hasattr(self, 'advanced_processor'):
            print("Starting advanced processing...")
            self.advanced_processor.start_recording()
            input("Press Enter to stop recording...")
            self.advanced_processor.stop_recording()
            print("Advanced processing complete")
            return True
        return False
    
    def _record_audio(self):
        stream = self.pyaudio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        while self.recording:
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            self.frames.append(data)
        
        stream.stop_stream()
        stream.close()
    
    def _record_video(self):
        while self.recording and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            face_info = self.face_detector.detect_face_direction(frame)
            
            self.video_frames.append((frame, face_info, time.time()))
            
            self.face_info_list.append(face_info)
            
            self._display_frame_with_face_info(frame, face_info)
            
            time.sleep(0.03)
    
    def _display_frame_with_face_info(self, frame, face_info):
        direction = face_info.get('direction', 'unknown')
        angle = face_info.get('angle')
        
        color = (0, 255, 0) if direction == 'towards' else (0, 0, 255)
        
        cv2.putText(frame, f"Direction: {direction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if angle is not None:
            cv2.putText(frame, f"Angle: {angle:.1f} degrees", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        is_spoken_to = face_info.get('is_spoken_to', False)
        status = "Spoken To" if is_spoken_to else "Not Spoken To"
        cv2.putText(frame, status, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Integrated Recording', frame)
        cv2.waitKey(1)
    
    def _save_audio(self, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
    
    def _summarize_face_info(self):
        if not self.face_info_list:
            return {"error": "No face information available"}
        
        directions = [info.get('direction', 'unknown') for info in self.face_info_list]
        direction_counts = {
            'towards': directions.count('towards'),
            'away': directions.count('away'),
            'unknown': directions.count('unknown')
        }
        
        total_frames = len(directions)
        towards_percentage = (direction_counts['towards'] / total_frames) * 100 if total_frames > 0 else 0
        
        spoken_to = [info.get('is_spoken_to', False) for info in self.face_info_list].count(True)
        spoken_to_percentage = (spoken_to / total_frames) * 100 if total_frames > 0 else 0
        
        return {
            'total_frames': total_frames,
            'direction_counts': direction_counts,
            'towards_percentage': towards_percentage,
            'spoken_to_percentage': spoken_to_percentage
        }

def main():
    parser = argparse.ArgumentParser(description="Integrated audio/video processor")
    parser.add_argument("--output-dir", type=str, default="integrated_results",
                       help="Directory to save processing results")
    parser.add_argument("--record-seconds", type=int, default=None,
                       help="Record for specified number of seconds (default: manual stop)")
    parser.add_argument("--no-webcam", action="store_true",
                       help="Don't use webcam for recording")
    parser.add_argument("--advanced", action="store_true",
                       help="Use advanced processing features")
    
    args = parser.parse_args()
    
    processor = IntegratedLaptopProcessor(
        output_dir=args.output_dir,
        use_webcam=not args.no_webcam
    )
    
    if args.advanced:
        processor.process_with_advanced_features()
    else:
        print("Starting recording...")
        processor.start_recording(record_seconds=args.record_seconds)
        
        if args.record_seconds is None:
            print("Press Enter to stop recording...")
            input()
            results = processor.stop_recording()
            print(f"Recording stopped. Transcript: {results.get('transcript', '')}")

if __name__ == "__main__":
    main()