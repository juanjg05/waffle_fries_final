#!/usr/bin/env python3
"""
Main entry point for audio-video processing application.
"""

import os
import logging
from src.audio_video_processing.scripts.simplified_laptop_processor import SimplifiedLaptopProcessor, main as processor_main

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data directories if they don't exist
    os.makedirs("data/conversation_data", exist_ok=True)
    os.makedirs("data/speaker_contexts", exist_ok=True)
    os.makedirs("src/models", exist_ok=True)
    
    # Run the processor
    processor_main()
