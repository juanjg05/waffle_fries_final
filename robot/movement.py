# possible turning angles

import numpy as np
import logging
from typing import Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotMovement:
    def __init__(self):
        self.current_position = np.array([0.0, 0.0])  # x, y coordinates
        self.current_angle = 0.0  # angle in radians
        self.max_speed = 0.5  # meters per second
        self.max_angular_speed = 1.0  # radians per second
        
    def move_toward_speaker(self, direction_info: Dict) -> None:
        """
        Move the robot toward the detected speaker.
        
        Args:
            direction_info (Dict): Dictionary containing direction information
                with keys 'angle' (in radians) and 'distance' (in meters)
        """
        try:
            target_angle = direction_info['angle']
            distance = direction_info['distance']
            
            # Calculate angle difference
            angle_diff = self._normalize_angle(target_angle - self.current_angle)
            
            # Rotate toward the speaker
            if abs(angle_diff) > 0.1:  # Threshold for rotation
                self._rotate(angle_diff)
            
            # Move forward if we're facing the right direction
            if abs(angle_diff) < 0.2:  # Threshold for forward movement
                self._move_forward(min(distance, self.max_speed))
                
            logger.info(f"Moving toward speaker: angle={target_angle:.2f}, distance={distance:.2f}")
            
        except Exception as e:
            logger.error(f"Error in move_toward_speaker: {str(e)}")
            raise
    
    def _rotate(self, angle: float) -> None:
        """
        Rotate the robot by the specified angle.
        
        Args:
            angle (float): Angle to rotate in radians
        """
        # Limit rotation speed
        rotation = np.clip(angle, -self.max_angular_speed, self.max_angular_speed)
        self.current_angle = self._normalize_angle(self.current_angle + rotation)
        logger.info(f"Rotated by {rotation:.2f} radians")
    
    def _move_forward(self, distance: float) -> None:
        """
        Move the robot forward by the specified distance.
        
        Args:
            distance (float): Distance to move in meters
        """
        # Update position based on current angle
        dx = distance * np.cos(self.current_angle)
        dy = distance * np.sin(self.current_angle)
        self.current_position += np.array([dx, dy])
        logger.info(f"Moved forward by {distance:.2f} meters")
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to be between -pi and pi.
        
        Args:
            angle (float): Input angle in radians
            
        Returns:
            float: Normalized angle
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

# Create a global instance
robot = RobotMovement()

def move_robot_toward_speaker(direction_info: Dict) -> None:
    """
    Global function to move the robot toward the speaker.
    
    Args:
        direction_info (Dict): Dictionary containing direction information
    """
    robot.move_toward_speaker(direction_info)