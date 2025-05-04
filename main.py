from models.diarization_model import diarize_speech, combine_diarization_with_transcript, DiarizationResult
from robot.movement import move_robot_toward_speaker
from utils.memory import SpeakerMemory
from robot.spatial_audio import get_speaker_direction
import whisper
import logging
import os
from typing import List, Tuple, Dict
import json
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio(audio_file: str) -> Tuple[str, List[Tuple[str, float, float]]]:
    """
    Transcribes the audio file using Whisper.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        Tuple[str, List[Tuple[str, float, float]]]: (transcription, word_timestamps)
    """
    try:
        # Load Whisper model
        model = whisper.load_model("base")

        # Perform transcription
        result = model.transcribe(audio_file)
        
        # Extract word timestamps
        word_timestamps = []
        for segment in result["segments"]:
            for word in segment["words"]:
                word_timestamps.append((word["word"], word["start"], word["end"]))
            
        return result["text"], word_timestamps
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def extract_voice_features(audio_data: np.ndarray, sr: int) -> Dict:
    """
    Extract voice features from audio data for speaker matching.
    
    Args:
        audio_data (np.ndarray): Audio data
        sr (int): Sample rate
        
    Returns:
        Dict: Voice features
    """
    # This is a placeholder - you would implement actual voice feature extraction
    # For example, using librosa to extract MFCCs, pitch, etc.
    return {
        'mfcc': [],  # Mel-frequency cepstral coefficients
        'pitch': [],  # Pitch contour
        'energy': []  # Energy contour
    }

def process_audio(
    audio_file: str
) -> List[Dict]:
    """
    Process an audio file through the complete pipeline.
    
    Args:
        audio_file (str): Path to the audio file
        
    Returns:
        List[Dict]: List of processed segments with speaker info
    """
    try:
        # Step 1: Get speaker direction
        direction_info = get_speaker_direction(audio_file)
        logger.info(f"Speaker direction: {direction_info}")
        
        # Step 2: Perform diarization
        diarization_results = diarize_speech(audio_file)
        logger.info(f"Found {len(diarization_results)} speaker segments")
        
        # Step 3: Transcribe the audio
        transcript, word_timestamps = transcribe_audio(audio_file)
        logger.info("Transcription completed")
        
        # Step 4: Combine diarization with transcript
        segments = combine_diarization_with_transcript(
            diarization_results,
            transcript,
            word_timestamps
        )
        
        # Step 5: Process each segment
        processed_segments = []
        for segment in segments:
            processed_segments.append({
                'speaker_id': segment.speaker_id,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'transcript': segment.transcript,
                'direction': direction_info
            })
            
        return processed_segments
        
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    audio_file = "path/to/audio.wav"
    segments = process_audio(audio_file)
    
    # Print results
    for segment in segments:
        print(f"Speaker {segment['speaker_id']}:")
        print(f"  Time: {segment['start_time']:.2f} - {segment['end_time']:.2f}")
        print(f"  Transcript: {segment['transcript']}")
        print(f"  Direction: {segment['direction']}")
