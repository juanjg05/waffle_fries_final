#!/usr/bin/env python3

import os
import logging
from src.audio_video_processing.scripts.simplified_laptop_processor import SimplifiedLaptopProcessor, main as processor_main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    os.makedirs("data/conversation_data", exist_ok=True)
    os.makedirs("data/speaker_contexts", exist_ok=True)
    os.makedirs("src/models", exist_ok=True)
    
    processor_main()
