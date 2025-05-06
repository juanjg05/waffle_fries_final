#!/bin/bash

# Set up environment for webcam and microphone access in WSL

echo "Setting up environment for audio/video processing in WSL..."

# Get Windows host IP from WSL resolv.conf
WINDOWS_HOST=$(grep nameserver /etc/resolv.conf | awk '{print $2}')
echo "Detected Windows host IP: $WINDOWS_HOST"

# Set up PulseAudio for microphone access
export PULSE_SERVER=tcp:$WINDOWS_HOST
echo "PULSE_SERVER set to tcp:$WINDOWS_HOST"

# Add to bashrc if not already there
if ! grep -q "PULSE_SERVER=tcp" ~/.bashrc; then
    echo "Adding PULSE_SERVER to ~/.bashrc..."
    echo "export PULSE_SERVER=tcp:\$(grep nameserver /etc/resolv.conf | awk '{print \$2}')" >> ~/.bashrc
    echo "Added. This will persist across terminal sessions."
fi

# Check for webcam
echo -e "\nChecking for webcams:"
ls -l /dev/video* 2>/dev/null
if [ $? -ne 0 ]; then
    echo "No webcam detected. Please follow the instructions in webcam_setup_instructions.md"
    echo "to share your webcam from Windows to WSL using USBIPD."
else
    echo "Webcam detected!"
fi

# Check for audio devices
echo -e "\nChecking for audio devices:"
pacmd list-sources 2>/dev/null
if [ $? -ne 0 ]; then
    echo "No PulseAudio server detected. Make sure PulseAudio is running on Windows."
    echo "Please follow the instructions in audio_setup_instructions.md."
else
    echo "PulseAudio connection successful!"
fi

echo -e "\nEnvironment setup complete."
echo "Next steps:"
echo "1. Verify webcam access: ./verify_webcam.py"
echo "2. Verify microphone access: ./verify_audio.py"
echo "3. Run the audio/video processor: python audio_video_processing/scripts/simplified_laptop_processor.py" 