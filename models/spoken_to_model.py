

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ProsodyFeatures:
    speaking_rate: float  # Words per second
    pitch_mean: float    # Mean pitch
    pitch_std: float     # Pitch standard deviation
    volume_mean: float   # Mean volume
    volume_std: float    # Volume standard deviation

@dataclass
class SpokenToFeatures:
    prosody: ProsodyFeatures
    speaker_angle: float
    speaker_distance: float
    num_speakers: int
    transcript: str

class IntentClassifier(nn.Module):
    def __init__(self, model_name: str = "prajjwal1/bert-tiny"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(pooled)

class SpokenToModel:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the spoken-to identification model.
        
        Args:
            model_path (Optional[str]): Path to a pre-trained model. If None, will use default weights.
        """
        self.intent_classifier = IntentClassifier()
        if model_path:
            self.intent_classifier.load_state_dict(torch.load(model_path))
        self.intent_classifier.eval()
        
        # Threshold for considering speech as directed to robot
        self.threshold = 0.7
        
    def extract_prosody_features(self, audio_data: np.ndarray, sr: int) -> ProsodyFeatures:
        """
        Extract prosody features from audio data.
        
        Args:
            audio_data (np.ndarray): Audio data
            sr (int): Sample rate
            
        Returns:
            ProsodyFeatures: Extracted prosody features
        """
        # This is a simplified version - in practice, you'd want more sophisticated
        # prosody analysis using libraries like librosa
        volume = np.abs(audio_data)
        
        return ProsodyFeatures(
            speaking_rate=0.0,  # Would need word timing information
            pitch_mean=0.0,     # Would need pitch tracking
            pitch_std=0.0,      # Would need pitch tracking
            volume_mean=float(np.mean(volume)),
            volume_std=float(np.std(volume))
        )
    
    def get_intent_score(self, transcript: str) -> float:
        """
        Get the intent score for a transcript using BERT-Tiny.
        
        Args:
            transcript (str): The transcript to analyze
            
        Returns:
            float: Intent score between 0 and 1
        """
        inputs = self.intent_classifier.tokenizer(
            transcript,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        
        with torch.no_grad():
            score = self.intent_classifier(
                inputs["input_ids"],
                inputs["attention_mask"]
            ).item()
            
        return score
    
    def is_spoken_to_robot(self, features: SpokenToFeatures) -> Tuple[bool, float]:
        """
        Determine if the speech is directed to the robot.
        
        Args:
            features (SpokenToFeatures): Features of the speech
            
        Returns:
            Tuple[bool, float]: (is_spoken_to_robot, confidence)
        """
        # Get intent score from BERT
        intent_score = self.get_intent_score(features.transcript)
        
        # Calculate spatial score based on angle and distance
        # Higher score when speaker is in front and close
        angle_score = np.cos(np.radians(features.speaker_angle))
        distance_score = 1 - features.speaker_distance
        spatial_score = (angle_score + distance_score) / 2
        
        # Calculate prosody score
        # Higher score for slower speech and higher pitch variation
        prosody_score = (
            (1 - features.prosody.speaking_rate) +  # Slower speech
            features.prosody.pitch_std +            # More pitch variation
            features.prosody.volume_std             # More volume variation
        ) / 3
        
        # Combine scores with weights
        final_score = (
            0.4 * intent_score +    # Intent from transcript
            0.3 * spatial_score +   # Spatial position
            0.3 * prosody_score     # Prosody features
        )
        
        return final_score > self.threshold, final_score

# Example usage:
if __name__ == "__main__":
    model = SpokenToModel()
    
    # Example features
    features = SpokenToFeatures(
        prosody=ProsodyFeatures(
            speaking_rate=0.5,
            pitch_mean=0.0,
            pitch_std=0.5,
            volume_mean=0.7,
            volume_std=0.3
        ),
        speaker_angle=0.0,  # Speaker is directly in front
        speaker_distance=0.3,  # Speaker is relatively close
        num_speakers=2,
        transcript="Hey robot, can you help me with something?"
    )
    
    is_spoken_to, confidence = model.is_spoken_to_robot(features)
    print(f"Is spoken to robot: {is_spoken_to} (confidence: {confidence:.2f})") 