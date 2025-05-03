from models.nemo_diarization_model import diarize_speech, combine_diarization_with_transcript, DiarizationResult
from utils.rttm_parser import parse_rttm
from robot.movement import move_robot_toward_speaker
from utils.memory import SpeakerMemory
from models.speaker_name_model import SpeakerNameModel
from models.spoken_to_model import SpokenToModel, SpokenToFeatures, ProsodyFeatures
from robot.spatial_audio import get_speaker_direction
import nemo.collections.asr as nemo_asr
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
    Transcribes the audio file using a pre-trained ASR model.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        Tuple[str, List[Tuple[str, float, float]]]: (transcription, word_timestamps)
    """
    try:
        # Load a pre-trained ASR model
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/quartznet15x5base-en")

        # Perform transcription with word timestamps
        transcription = asr_model.transcribe(
            [audio_file],
            return_hypotheses=True,
            batch_size=1
        )
        
        # Extract word timestamps
        word_timestamps = []
        for word in transcription[0].words:
            word_timestamps.append((word.word, word.start_time, word.end_time))
            
        return transcription[0].text, word_timestamps
        
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
    audio_file: str,
    speaker_name_model: SpeakerNameModel,
    spoken_to_model: SpokenToModel
) -> List[Dict]:
    """
    Process an audio file through the complete pipeline.
    
    Args:
        audio_file (str): Path to the audio file
        speaker_name_model (SpeakerNameModel): Initialized speaker name model
        spoken_to_model (SpokenToModel): Initialized spoken-to model
        
    Returns:
        List[Dict]: List of processed segments with speaker info and robot-directed status
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
            # Extract voice features for this segment
            # Note: In practice, you'd need to extract the audio segment first
            voice_features = extract_voice_features(None, None)  # Placeholder
            
            # Get speaker name
            name = speaker_name_model.update_speaker(
                segment.speaker_id,
                segment.transcript,
                voice_features
            )
            
            # Create features for spoken-to detection
            features = SpokenToFeatures(
                prosody=ProsodyFeatures(
                    speaking_rate=0.0,  # Would need to calculate from word timestamps
                    pitch_mean=0.0,     # Would need pitch analysis
                    pitch_std=0.0,      # Would need pitch analysis
                    volume_mean=0.0,    # Would need volume analysis
                    volume_std=0.0      # Would need volume analysis
                ),
                speaker_angle=direction_info['angle'],
                speaker_distance=direction_info['distance'],
                num_speakers=len(diarization_results),
                transcript=segment.transcript
            )
            
            # Check if spoken to robot
            is_spoken_to, confidence = spoken_to_model.is_spoken_to_robot(features)
            
            processed_segments.append({
                'speaker_id': segment.speaker_id,
                'name': name,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'transcript': segment.transcript,
                'is_spoken_to_robot': is_spoken_to,
                'confidence': confidence,
                'direction': direction_info
            })
            
        return processed_segments
        
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise

def main():
    # Initialize models
    speaker_name_model = SpeakerNameModel()  # No API key needed now
    spoken_to_model = SpokenToModel()
    
    # Process audio file
    audio_file = "data/audio/sample_audio.wav"
    
    try:
        segments = process_audio(audio_file, speaker_name_model, spoken_to_model)
        
        # Print results
        for segment in segments:
            print(f"\nSpeaker {segment['name'] or segment['speaker_id']}:")
            print(f"Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
            print(f"Transcript: {segment['transcript']}")
            print(f"Spoken to robot: {segment['is_spoken_to_robot']} (confidence: {segment['confidence']:.2f})")
            print(f"Direction: {segment['direction']}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
