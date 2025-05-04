#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Any
import json

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_face_direction(self, frame: np.ndarray) -> Tuple[str, float, Optional[float], Optional[Any]]:
        """
        Detect face direction using MediaPipe Face Mesh.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple containing:
            - Face direction ('towards' or 'away')
            - Face angle in degrees
            - Confidence score
            - Landmarks for visualization
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return 'unknown', 0.0, None, None
            
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get nose bridge landmarks (indices 1 and 4)
        nose_bridge = [
            face_landmarks.landmark[1],  # Top of nose bridge
            face_landmarks.landmark[4]   # Bottom of nose bridge
        ]
        
        # Calculate angle
        dx = nose_bridge[1].x - nose_bridge[0].x
        dy = nose_bridge[1].y - nose_bridge[0].y
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Determine direction
        if abs(angle) < 30:  # Threshold for "towards"
            direction = 'towards'
        else:
            direction = 'away'
            
        return direction, abs(angle), 0.9, face_landmarks

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.face_info_pub = self.create_publisher(
            String,
            'face_info',
            10
        )
        
        self.get_logger().info('Face detection node initialized')
        
    def image_callback(self, msg: Image):
        """
        Process incoming image messages
        """
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect face direction
            direction, angle, confidence, landmarks = self.face_detector.detect_face_direction(frame)
            
            # Publish face info
            face_info = {
                'direction': direction,
                'angle': float(angle),
                'confidence': float(confidence) if confidence else None
            }
            
            msg = String()
            msg.data = json.dumps(face_info)
            self.face_info_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 