
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Audio
from models.nemo_diarization_model import diarize_speech, combine_diarization_with_transcript, DiarizationResult
from utils.rttm_parser import parse_rttm
from robot.movement import move_robot_toward_speaker
from utils.memory import SpeakerMemory
from robot.spatial_audio import get_speaker_direction
import nemo.collections.asr as nemo_asr
import logging
import os
from typing import List, Tuple, Dict
import json
import numpy as np

class AudioProcessorNode(Node):
    def __init__(self):
        super().__init__('audio_processor_node')
        
        # Initialize components
        self.memory = SpeakerMemory()
        
        # Create subscribers and publishers
        self.audio_sub = self.create_subscription(
            Audio,
            'audio_input',
            self.audio_callback,
            10
        )
        
        self.speaker_pub = self.create_publisher(
            SpeakerInfo,
            'speaker_info',
            10
        )

    def audio_callback(self, msg: Audio):
        """
        Process incoming audio messages
        """
        try:
            # Convert audio message to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.float32)
            
            # Get speaker direction
            direction_info = get_speaker_direction(audio_data)
            
            # Perform diarization
            diarization_results = diarize_speech(audio_data)
            
            # Transcribe audio
            transcript, word_timestamps = self.transcribe_audio(audio_data)
            
            # Combine diarization with transcript
            segments = combine_diarization_with_transcript(
                diarization_results,
                transcript,
                word_timestamps
            )
            
            # Process each segment
            for segment in segments:
                # Publish speaker info
                speaker_info = {
                    'speaker_id': segment.speaker_id,
                    'transcript': segment.transcript,
                    'direction': direction_info
                }
                self.publish_speaker_info(speaker_info)
                
        except Exception as e:
            self.get_logger().error(f"Error processing audio: {str(e)}")

    def transcribe_audio(self, audio_data: np.ndarray) -> Tuple[str, List[Tuple[str, float, float]]]:
        """
        Transcribe audio data using the ASR model
        """
        try:
            transcription = self.asr_model.transcribe(
                [audio_data],
                return_hypotheses=True,
                batch_size=1
            )
            
            word_timestamps = []
            for word in transcription[0].words:
                word_timestamps.append((word.word, word.start_time, word.end_time))
                
            return transcription[0].text, word_timestamps
            
        except Exception as e:
            self.get_logger().error(f'Transcription failed: {str(e)}')
            raise

    def extract_voice_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """
        Extract voice features from audio data
        """
        # Placeholder - implement actual feature extraction
        return {
            'mfcc': [],
            'pitch': [],
            'energy': []
        }

    def move_robot_toward_speaker(self, direction_info: Dict):
        """
        Move robot towards the detected speaker
        """
        cmd = Twist()
        
        # Set linear velocity based on distance
        cmd.linear.x = min(0.5, direction_info['distance'] * 0.1)
        
        # Set angular velocity based on angle
        cmd.angular.z = direction_info['angle'] * 0.1
        
        self.robot_cmd_pub.publish(cmd)

    def publish_speaker_info(self, info: Dict):
        """
        Publish speaker information
        """
        msg = String()
        msg.data = str(info)
        self.speaker_info_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AudioProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 