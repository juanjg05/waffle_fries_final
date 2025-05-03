import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os
import time

def record_audio(duration=10, sample_rate=16000, output_file="tests/data/recorded_test.wav"):
    """
    Record audio for the specified duration.
    
    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Sample rate for recording
        output_file (str): Path to save the recorded audio
    """
    print(f"Recording will start in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording started...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the recording
    wav.write(output_file, sample_rate, recording)
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    # Record 10 seconds of audio
    record_audio(duration=10) 