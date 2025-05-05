#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ByteMultiArray
from geometry_msgs.msg import Twist
from robot_audio_processor.models.diarization_model import diarize_speech, DiarizationResult
from robot.movement import move_robot_toward_speaker
from utils.memory import SpeakerMemory
from robot.spatial_audio import get_speaker_direction
import whisper
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
            ByteMultiArray,
            'audio_input',
            self.audio_callback,
            10
        )
        
        self.speaker_pub = self.create_publisher(
            String,
            'speaker_info',
            10
        )

    def audio_callback(self, msg: ByteMultiArray):
        """
        Process incoming audio messages
        """
        try:
            # Convert ByteMultiArray to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.float32)
            
            # Get speaker direction
            direction_info = get_speaker_direction(audio_data)
            
            # Perform diarization
            diarization_results = diarize_speech(audio_data)
            
            # Transcribe audio
            transcript, word_timestamps = self.transcribe_audio(audio_data)
            
            # Process each segment
            for segment in diarization_results:
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
        Transcribe audio data using Whisper
        """
        try:
            # Load Whisper model
            model = whisper.load_model("base")

            # Perform transcription
            result = model.transcribe(audio_data)
            
            # Extract word timestamps
            word_timestamps = []
            for segment in result["segments"]:
                for word in segment["words"]:
                    word_timestamps.append((word["word"], word["start"], word["end"]))
                
            return result["text"], word_timestamps
            
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
        msg.data = json.dumps(info)
        self.speaker_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AudioProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 