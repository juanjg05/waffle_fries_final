"""
Robot Audio Processor package for ROS2.
This package handles audio processing, speaker diarization, and face detection.
"""

from .models.diarization_model import diarize_speech, DiarizationResult
from .utils.memory import SpeakerMemory
from .utils.spatial_audio import get_speaker_direction

__all__ = [
    'diarize_speech',
    'DiarizationResult',
    'SpeakerMemory',
    'get_speaker_direction'
]

__version__ = '0.1.0' 