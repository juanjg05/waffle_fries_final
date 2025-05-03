# 

from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpeakerInfo:
    speaker_id: str
    name: Optional[str] = None
    conversations: List[str] = None
    confidence: float = 0.0
    voice_features: Optional[Dict] = None  # Store voice features for future matching

    def __post_init__(self):
        if self.conversations is None:
            self.conversations = []
        if self.voice_features is None:
            self.voice_features = {}

class SpeakerNameModel:
    def __init__(self, speakers_file: Optional[str] = "data/speakers.json"):
        """
        Initialize the speaker name model with a file to persist speaker information.
        
        Args:
            speakers_file (Optional[str]): Path to JSON file storing speaker information
        """
        self.speakers_file = speakers_file
        self.speakers: Dict[str, SpeakerInfo] = {}
        self._load_speakers()
        
    def _load_speakers(self):
        """Load speaker information from the JSON file."""
        if os.path.exists(self.speakers_file):
            try:
                with open(self.speakers_file, 'r') as f:
                    data = json.load(f)
                    for speaker_id, info in data.items():
                        self.speakers[speaker_id] = SpeakerInfo(
                            speaker_id=speaker_id,
                            name=info.get('name'),
                            conversations=info.get('conversations', []),
                            confidence=info.get('confidence', 0.0),
                            voice_features=info.get('voice_features', {})
                        )
            except Exception as e:
                logger.error(f"Error loading speakers file: {e}")
                # Start with empty speakers if file is corrupted
                self.speakers = {}
    
    def _save_speakers(self):
        """Save speaker information to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self.speakers_file), exist_ok=True)
            with open(self.speakers_file, 'w') as f:
                data = {
                    speaker_id: {
                        'name': info.name,
                        'conversations': info.conversations,
                        'confidence': info.confidence,
                        'voice_features': info.voice_features
                    }
                    for speaker_id, info in self.speakers.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving speakers file: {e}")

    def update_speaker(self, speaker_id: str, transcript: str, voice_features: Optional[Dict] = None) -> Optional[str]:
        """
        Update a speaker's information and try to identify their name from the transcript.
        
        Args:
            speaker_id (str): The speaker's ID from NeMo
            transcript (str): The transcript to analyze for name information
            voice_features (Optional[Dict]): Voice features for speaker matching
            
        Returns:
            Optional[str]: The speaker's name if identified, None otherwise
        """
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = SpeakerInfo(speaker_id=speaker_id)
            
        speaker = self.speakers[speaker_id]
        speaker.conversations.append(transcript)
        
        # Update voice features if provided
        if voice_features:
            speaker.voice_features.update(voice_features)
        
        # Try to extract name from transcript if we don't have one
        if not speaker.name:
            # Look for common name patterns in the transcript
            # This is a simple implementation - you might want to use more sophisticated NLP
            name_candidates = self._extract_name_candidates(transcript)
            if name_candidates:
                speaker.name = name_candidates[0]  # Take the first candidate
                speaker.confidence = 0.7  # Moderate confidence for name extraction
                self._save_speakers()
                return speaker.name
                
        return speaker.name

    def _extract_name_candidates(self, transcript: str) -> List[str]:
        """
        Extract potential names from a transcript.
        This is a simple implementation - you might want to use more sophisticated NLP.
        
        Args:
            transcript (str): The transcript to analyze
            
        Returns:
            List[str]: List of potential names found in the transcript
        """
        # Common name introduction patterns
        patterns = [
            "I'm", "I am", "My name is", "This is", "Call me"
        ]
        
        candidates = []
        words = transcript.split()
        
        for i, word in enumerate(words):
            # Check for introduction patterns
            if word in patterns and i + 1 < len(words):
                # Take the next word as a potential name
                potential_name = words[i + 1].strip('.,!?')
                if potential_name and potential_name[0].isupper():
                    candidates.append(potential_name)
        
        return candidates

    def get_speaker_info(self, speaker_id: str) -> Optional[SpeakerInfo]:
        """
        Get information about a speaker.
        
        Args:
            speaker_id (str): The speaker's ID
            
        Returns:
            Optional[SpeakerInfo]: Speaker information if found, None otherwise
        """
        return self.speakers.get(speaker_id)

    def get_all_speakers(self) -> Dict[str, SpeakerInfo]:
        """
        Get information about all known speakers.
        
        Returns:
            Dict[str, SpeakerInfo]: Dictionary of all speaker information
        """
        return self.speakers

    def match_speaker(self, voice_features: Dict) -> Optional[str]:
        """
        Try to match voice features to a known speaker.
        
        Args:
            voice_features (Dict): Voice features to match
            
        Returns:
            Optional[str]: Speaker ID if matched, None otherwise
        """
        # This is a placeholder - you would implement actual voice feature matching
        # For example, using cosine similarity between voice embeddings
        return None

# Example usage:
if __name__ == "__main__":
    model = SpeakerNameModel()
    
    # Example conversations with NeMo speaker IDs
    conversations = [
        ("SPEAKER_1", "Hi, I'm John and I'll be your guide today."),
        ("SPEAKER_2", "My name is Sarah, nice to meet you all."),
        ("SPEAKER_1", "As I mentioned, we'll be exploring the facility.")
    ]
    
    # Update speakers with conversations
    for speaker_id, transcript in conversations:
        name = model.update_speaker(speaker_id, transcript)
        if name:
            print(f"Identified speaker {speaker_id} as {name}")
