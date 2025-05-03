import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UtteranceSegment:
    def __init__(self, speaker_id: str, start_time: float, end_time: float, 
                 transcript: str = "", confidence: float = 1.0):
        self.speaker_id = speaker_id
        self.start_time = start_time
        self.end_time = end_time
        self.transcript = transcript
        self.confidence = confidence
        self.is_end_of_utterance = False  # Flag for end of utterance

class ConversationParser:
    def __init__(self, silence_threshold: float = 1.0, 
                 min_utterance_duration: float = 0.5,
                 max_utterance_duration: float = 10.0):
        """
        Initialize the conversation parser.
        
        Args:
            silence_threshold: Time in seconds to consider as end of utterance
            min_utterance_duration: Minimum duration for a valid utterance
            max_utterance_duration: Maximum duration for a single utterance
        """
        self.silence_threshold = silence_threshold
        self.min_utterance_duration = min_utterance_duration
        self.max_utterance_duration = max_utterance_duration

    def parse_rttm(self, rttm_file: str) -> List[UtteranceSegment]:
        """
        Parse RTTM file to get speaker segments.
        """
        segments = []
        try:
            with open(rttm_file, 'r') as file:
                for line in file:
                    fields = line.strip().split()
                    if fields[0] == 'SPEAKER':
                        start_time = float(fields[3])
                        duration = float(fields[4])
                        speaker_id = fields[7]
                        
                        # Skip segments that are too short
                        if duration < self.min_utterance_duration:
                            continue
                            
                        # Split long segments
                        if duration > self.max_utterance_duration:
                            num_splits = int(np.ceil(duration / self.max_utterance_duration))
                            split_duration = duration / num_splits
                            for i in range(num_splits):
                                split_start = start_time + (i * split_duration)
                                segments.append(UtteranceSegment(
                                    speaker_id=speaker_id,
                                    start_time=split_start,
                                    end_time=split_start + split_duration
                                ))
                        else:
                            segments.append(UtteranceSegment(
                                speaker_id=speaker_id,
                                start_time=start_time,
                                end_time=start_time + duration
                            ))
        except Exception as e:
            logger.error(f"Error parsing RTTM file: {str(e)}")
            raise

        return self._mark_end_of_utterances(segments)

    def parse_transcript(self, transcript_file: str) -> List[Dict]:
        """
        Parse transcript file to get word-level timestamps.
        """
        word_segments = []
        try:
            with open(transcript_file, 'r') as file:
                for line in file:
                    fields = line.strip().split()
                    if len(fields) >= 4:
                        start_time = float(fields[0])
                        end_time = float(fields[1])
                        word = fields[3]
                        word_segments.append({
                            'word': word,
                            'start_time': start_time,
                            'end_time': end_time
                        })
        except Exception as e:
            logger.error(f"Error parsing transcript file: {str(e)}")
            raise

        return word_segments

    def _mark_end_of_utterances(self, segments: List[UtteranceSegment]) -> List[UtteranceSegment]:
        """
        Mark segments that are likely end of utterances based on silence threshold.
        """
        if not segments:
            return segments

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        
        # Mark end of utterances based on silence threshold
        for i in range(len(sorted_segments) - 1):
            current_segment = sorted_segments[i]
            next_segment = sorted_segments[i + 1]
            
            # If there's a significant gap between segments
            if (next_segment.start_time - current_segment.end_time) > self.silence_threshold:
                current_segment.is_end_of_utterance = True
        
        # Mark the last segment as end of utterance
        sorted_segments[-1].is_end_of_utterance = True
        
        return sorted_segments

    def combine_segments_with_transcript(
        self, 
        segments: List[UtteranceSegment],
        word_segments: List[Dict]
    ) -> List[UtteranceSegment]:
        """
        Combine speaker segments with transcript words.
        """
        for segment in segments:
            segment_words = []
            for word_seg in word_segments:
                # Check if word falls within segment time
                if (word_seg['start_time'] >= segment.start_time and 
                    word_seg['end_time'] <= segment.end_time):
                    segment_words.append(word_seg['word'])
            
            segment.transcript = " ".join(segment_words)
        
        return segments

    def prepare_for_bert(self, segments: List[UtteranceSegment]) -> List[Dict]:
        """
        Prepare segments for BERT-tiny processing with end-of-utterance tokens.
        """
        bert_inputs = []
        current_speaker = None
        current_text = []
        
        for segment in segments:
            if not segment.transcript:  # Skip segments without transcript
                continue
                
            if current_speaker != segment.speaker_id:
                # If we have accumulated text, add it to bert_inputs
                if current_text:
                    bert_inputs.append({
                        'speaker_id': current_speaker,
                        'text': ' '.join(current_text),
                        'is_end_of_utterance': True  # Previous speaker's utterance ends
                    })
                # Start new speaker
                current_speaker = segment.speaker_id
                current_text = [segment.transcript]
            else:
                # Add to current speaker's text
                current_text.append(segment.transcript)
            
            # If this is marked as end of utterance, add it to bert_inputs
            if segment.is_end_of_utterance:
                bert_inputs.append({
                    'speaker_id': current_speaker,
                    'text': ' '.join(current_text),
                    'is_end_of_utterance': True
                })
                current_text = []
        
        # Add any remaining text
        if current_text:
            bert_inputs.append({
                'speaker_id': current_speaker,
                'text': ' '.join(current_text),
                'is_end_of_utterance': True
            })
        
        return bert_inputs

def parse_conversation(rttm_file: str, transcript_file: str) -> List[Dict]:
    """
    Main function to parse conversation files and prepare for BERT-tiny.
    
    Args:
        rttm_file: Path to RTTM file
        transcript_file: Path to transcript file
        
    Returns:
        List of dictionaries ready for BERT-tiny processing
    """
    parser = ConversationParser()
    
    try:
        # Parse RTTM file
        segments = parser.parse_rttm(rttm_file)
        
        # Parse transcript file
        word_segments = parser.parse_transcript(transcript_file)
        
        # Combine segments with transcript
        segments_with_transcript = parser.combine_segments_with_transcript(
            segments, word_segments)
        
        # Prepare for BERT-tiny
        bert_inputs = parser.prepare_for_bert(segments_with_transcript)
        
        return bert_inputs
        
    except Exception as e:
        logger.error(f"Error processing conversation files: {str(e)}")
        raise