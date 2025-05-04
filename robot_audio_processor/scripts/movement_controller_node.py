#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json
from typing import Optional, Dict
import numpy as np

class MovementControllerNode(Node):
    def __init__(self):
        super().__init__('movement_controller_node')
        
        # Initialize state
        self.current_speaker = None
        self.current_face_direction = None
        self.last_face_angle = 0.0
        
        # Create subscribers
        self.face_sub = self.create_subscription(
            String,
            'face_info',
            self.face_callback,
            10
        )
        
        self.speaker_sub = self.create_subscription(
            String,
            'speaker_info',
            self.speaker_callback,
            10
        )
        
        # Create publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        self.get_logger().info('Movement controller node initialized')
        
    def face_callback(self, msg: String):
        """
        Process face detection information
        """
        try:
            face_info = json.loads(msg.data)
            self.current_face_direction = face_info['direction']
            self.last_face_angle = face_info['angle']
            
            # If face is detected and looking away, turn towards them
            if self.current_face_direction == 'away':
                self.turn_towards_speaker()
                
        except Exception as e:
            self.get_logger().error(f'Error processing face info: {str(e)}')
            
    def speaker_callback(self, msg: String):
        """
        Process speaker information
        """
        try:
            speaker_info = json.loads(msg.data)
            self.current_speaker = speaker_info['speaker_id']
            
            # If someone is speaking and not looking at the robot, turn towards them
            if self.current_speaker and self.current_face_direction == 'away':
                self.turn_towards_speaker()
                
        except Exception as e:
            self.get_logger().error(f'Error processing speaker info: {str(e)}')
            
    def turn_towards_speaker(self):
        """
        Turn the robot towards the detected speaker
        """
        cmd = Twist()
        
        # Calculate angular velocity based on face angle
        # Positive angle means turn right, negative means turn left
        angular_vel = np.clip(self.last_face_angle * 0.1, -0.5, 0.5)
        cmd.angular.z = angular_vel
        
        # Small forward velocity to keep moving towards the speaker
        cmd.linear.x = 0.1
        
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info(f'Turning towards speaker with angular velocity: {angular_vel}')

def main(args=None):
    rclpy.init(args=args)
    node = MovementControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 