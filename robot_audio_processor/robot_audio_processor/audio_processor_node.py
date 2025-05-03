#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Audio
from models.nemo_diarization_model import diarize_speech, combine_diarization_with_transcript
from models.speaker_name_model import SpeakerNameModel
from models.spoken_to_model import SpokenToModel, SpokenToFeatures, ProsodyFeatures
from robot.spatial_audio import get_speaker_direction
import nemo.collections.asr as nemo_asr
import logging
import numpy as np
from typing import List, Tuple, Dict

class AudioProcessorNode(Node):
    def __init__(self):
        super().__init__('audio_processor_node')
        
        # Initialize models
        self.speaker_name_model = SpeakerNameModel()
        self.spoken_to_model = SpokenToModel()
        self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/quartznet15x5base-en")
        
        # Create subscribers
        self.audio_sub = self.create_subscription(
            Audio,
            'audio_input',
            self.audio_callback,
            10)
            
        # Create publishers
        self.robot_cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10)
            
        self.speaker_info_pub = self.create_publisher(
            String,
            'speaker_info',
            10)
            
        self.get_logger().info('Audio processor node initialized')

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
                # Extract voice features
                voice_features = self.extract_voice_features(audio_data, msg.sample_rate)
                
                # Get speaker name
                name = self.speaker_name_model.update_speaker(
                    segment.speaker_id,
                    segment.transcript,
                    voice_features
                )
                
                # Create features for spoken-to detection
                features = SpokenToFeatures(
                    prosody=ProsodyFeatures(
                        speaking_rate=0.0,
                        pitch_mean=0.0,
                        pitch_std=0.0,
                        volume_mean=0.0,
                        volume_std=0.0
                    ),
                    speaker_angle=direction_info['angle'],
                    speaker_distance=direction_info['distance'],
                    num_speakers=len(diarization_results),
                    transcript=segment.transcript
                )
                
                # Check if spoken to robot
                is_spoken_to, confidence = self.spoken_to_model.is_spoken_to_robot(features)
                
                # If spoken to robot, move towards speaker
                if is_spoken_to and confidence > 0.7:
                    self.move_robot_toward_speaker(direction_info)
                
                # Publish speaker info
                speaker_info = {
                    'speaker_id': segment.speaker_id,
                    'name': name,
                    'transcript': segment.transcript,
                    'is_spoken_to_robot': is_spoken_to,
                    'confidence': confidence,
                    'direction': direction_info
                }
                self.publish_speaker_info(speaker_info)
                
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {str(e)}')

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