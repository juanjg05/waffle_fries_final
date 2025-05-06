#!/usr/bin/env python3

import cv2
import time
import sys

def verify_webcam():
    """Verify webcam access by capturing a few frames and displaying info."""
    print("Attempting to access webcam...")
    
    # List available camera devices
    print("\nListing available camera devices:")
    for i in range(5):  # Check the first 5 possible camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i} is available")
            # Get some camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  Resolution: {width}x{height}, FPS: {fps}")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"  Successfully read a frame of shape {frame.shape}")
            else:
                print(f"  Could not read a frame")
            
            cap.release()
        else:
            print(f"Camera index {i} is not available")
    
    # Try to access the default camera (usually index 0)
    print("\nAttempting to open default camera (index 0):")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open webcam! Please check the webcam connection.")
        return False
    
    # Try to capture 5 frames
    print("Webcam opened successfully. Attempting to capture 5 frames:")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: Shape {frame.shape}, Type {frame.dtype}")
        else:
            print(f"Failed to capture frame {i+1}")
            
        time.sleep(0.5)  # Small delay between frames
    
    # Release the camera
    cap.release()
    print("\nWebcam test completed!")
    return True

if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)
    verify_webcam() 