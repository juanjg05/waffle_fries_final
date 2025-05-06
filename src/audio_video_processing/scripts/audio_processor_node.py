#!/usr/bin/env python3

import numpy as np
import whisper
import os
from typing import List, Tuple, Dict
import json
import time
from datetime import datetime
import torch

class AudioProcessor:
    def __init__(self, output_dir="processed_audio"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.asr_model = whisper.load_model("base")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.asr_model = None
        
        self.current_audio_file = None
        self.results = []
    
    def process_audio_file(self, audio_file: str) -> Dict:
        try:
            start_time = time.time()
            self.current_audio_file = audio_file
            
            transcript, word_timestamps = self.transcribe_audio(audio_file)
            
            result = {
                'filename': os.path.basename(audio_file),
                'processed_at': datetime.now().isoformat(),
                'transcript': transcript,
                'word_timestamps': word_timestamps,
                'processing_time': time.time() - start_time
            }
            
            result_filename = os.path.join(
                self.output_dir, 
                os.path.splitext(os.path.basename(audio_file))[0] + "_transcript.json"
            )
            
            with open(result_filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return {
                'error': str(e),
                'filename': os.path.basename(audio_file) if audio_file else None
            }

    def transcribe_audio(self, audio_file: str) -> Tuple[str, List[Dict]]:
        try:
            if self.asr_model is None:
                print("Whisper model not loaded, cannot transcribe")
                return "", []
            
            result = self.asr_model.transcribe(audio_file, word_timestamps=True)
            
            transcript = result.get("text", "")
            
            word_timestamps = []
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    word_timestamps.append({
                        'word': word.get('word', ''),
                        'start': word.get('start', 0.0),
                        'end': word.get('end', 0.0),
                        'confidence': word.get('confidence', 0.0)
                    })
            
            return transcript, word_timestamps
            
        except Exception as e:
            print(f'Transcription failed: {str(e)}')
            return "", []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files with Whisper")
    parser.add_argument("--audio-file", type=str, help="Path to audio file to process")
    parser.add_argument("--output-dir", type=str, default="processed_audio", 
                        help="Directory to save processed results")
    
    args = parser.parse_args()
    
    processor = AudioProcessor(output_dir=args.output_dir)
    
    if args.audio_file:
        if os.path.exists(args.audio_file):
            result = processor.process_audio_file(args.audio_file)
            print(f"Transcript: {result.get('transcript', '')}")
        else:
            print(f"Error: Audio file not found: {args.audio_file}")
    else:
        print("No audio file specified. Please provide an audio file with --audio-file.")
        print("Example: python audio_processor_node.py --audio-file recording.wav")

if __name__ == "__main__":
    main() 