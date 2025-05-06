#!/usr/bin/env python3

import pyaudio
import time
import wave
import numpy as np
import os

def verify_audio():
    """Verify microphone access by recording a few seconds of audio."""
    print("Attempting to access microphone...")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # List available audio devices
    print("\nListing available audio devices:")
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Input Device id {i} - {device_info.get('name')}")
    
    # Get default input device info
    try:
        default_device_info = p.get_default_input_device_info()
        print(f"\nDefault input device: {default_device_info['name']} (index {default_device_info['index']})")
    except IOError:
        print("No default input device found.")
        return False
    
    # Set recording parameters
    format = pyaudio.paInt16
    channels = 1
    rate = 16000  # Common sample rate
    chunk = 1024
    seconds = 3
    
    # Try to open stream with default device
    try:
        print(f"\nAttempting to record {seconds} seconds of audio...")
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
        
        frames = []
        
        # Record for a few seconds
        for i in range(0, int(rate / chunk * seconds)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            
            # Print audio level for visualization
            audio_data = np.frombuffer(data, dtype=np.int16)
            level = np.abs(audio_data).mean()
            level_bar = '#' * int(level / 100)
            print(f"\rAudio level: {level:.2f} {level_bar}", end='')
        
        print("\n\nRecording complete!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Save the recorded audio to a file
        filename = "test_recording.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Saved recording to {filename}")
        print(f"File size: {os.path.getsize(filename)} bytes")
        print(f"Recording duration: {seconds} seconds")
        
    except Exception as e:
        print(f"Error while recording: {str(e)}")
        p.terminate()
        return False

    # Terminate PyAudio
    p.terminate()
    return True

if __name__ == "__main__":
    verify_audio() 