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
        
        # Movement parameters based on segway_rmp_ros2
        self.max_linear_vel = 0.75  # m/s
        self.max_angular_vel = 28.65  # deg/s
        self.linear_accel_limit = 0.1  # m/s^2
        self.angular_accel_limit = 1.0  # deg/s^2
        
        # Current velocities
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        
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
        
        # Create timer for smooth movement updates
        self.movement_timer = self.create_timer(
            0.05,  # 20Hz update rate
            self.update_movement
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
        Set target velocities for turning towards the speaker
        """
        # Convert angle to radians for angular velocity calculation
        angle_rad = np.radians(self.last_face_angle)
        
        # Calculate target angular velocity (proportional control)
        target_angular_vel = -angle_rad * 0.5  # Negative because positive angle means turn right
        
        # Limit angular velocity
        self.current_angular_vel = np.clip(
            target_angular_vel,
            -self.max_angular_vel,
            self.max_angular_vel
        )
        
        # Set small forward velocity when turning
        self.current_linear_vel = 0.1
        
        self.get_logger().info(f'Target angular velocity: {self.current_angular_vel:.2f} rad/s')
    
    def update_movement(self):
        """
        Update movement commands with smooth acceleration
        """
        cmd = Twist()
        
        # Apply acceleration limits
        if self.current_linear_vel > 0:
            cmd.linear.x = min(self.current_linear_vel, self.linear_accel_limit)
        else:
            cmd.linear.x = max(self.current_linear_vel, -self.linear_accel_limit)
            
        if self.current_angular_vel > 0:
            cmd.angular.z = min(self.current_angular_vel, self.angular_accel_limit)
        else:
            cmd.angular.z = max(self.current_angular_vel, -self.angular_accel_limit)
        
        # Publish movement command
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MovementControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 