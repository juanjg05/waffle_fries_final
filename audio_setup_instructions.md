# Microphone Setup Instructions for WSL

This guide will help you set up microphone access in WSL for audio processing.

## Prerequisites
- Windows 11 with WSL2
- PulseAudio server on Windows (we'll set this up)

## Steps to Set Up Microphone Access

### 1. Install PulseAudio on Windows
1. Download PulseAudio for Windows from the official site or GitHub mirror:
   https://www.freedesktop.org/wiki/Software/PulseAudio/
   
   (Alternatively, you can install it using Chocolatey if you have it installed:)
   ```
   choco install pulseaudio
   ```

2. Extract the PulseAudio files to a location on your Windows drive (e.g., `C:\PulseAudio`)

### 2. Configure PulseAudio for WSL
1. Create or modify the PulseAudio config file:
   - Open `C:\PulseAudio\etc\pulse\default.pa` in a text editor
   - Add these lines at the end:
     ```
     load-module module-native-protocol-tcp auth-anonymous=1
     load-module module-esound-protocol-tcp auth-anonymous=1
     load-module module-waveout sink_name=output source_name=input
     ```

2. Create or modify the PulseAudio daemon config:
   - Open `C:\PulseAudio\etc\pulse\daemon.conf` in a text editor
   - Add or modify these lines:
     ```
     exit-idle-time = -1
     ```

### 3. Start the PulseAudio Server on Windows
1. Open Command Prompt and navigate to your PulseAudio directory:
   ```
   cd C:\PulseAudio
   ```

2. Start the PulseAudio server with:
   ```
   pulseaudio.exe --exit-idle-time=-1 --verbose
   ```
   
   Keep this window open while using the microphone in WSL.

### 4. Configure WSL to Use Windows PulseAudio
1. In your WSL terminal, install PulseAudio client:
   ```
   sudo apt-get install -y pulseaudio-utils
   ```

2. Set the PULSE_SERVER environment variable to point to Windows:
   ```
   export PULSE_SERVER=tcp:$(grep nameserver /etc/resolv.conf | awk '{print $2}')
   ```

3. Add the above line to your ~/.bashrc to make it persistent:
   ```
   echo 'export PULSE_SERVER=tcp:$(grep nameserver /etc/resolv.conf | awk '{print $2}')' >> ~/.bashrc
   ```

### 5. Verify Microphone Access
1. In your WSL terminal, run our verification script:
   ```
   ./verify_audio.py
   ```
   This will check if the microphone is accessible through PyAudio.

### Troubleshooting

If the microphone is not detected:
1. Make sure the PulseAudio server is running on Windows
2. Check that the Windows firewall isn't blocking the connection
3. Try setting the PULSE_SERVER variable manually with your Windows IP:
   ```
   export PULSE_SERVER=tcp:192.168.x.x
   ```
4. Restart your WSL terminal after making changes

### Notes
- The PulseAudio server must be running on Windows whenever you need to use the microphone in WSL
- You may need to restart the server after system sleep/hibernation

## Running the Audio-Video Processing Project
Once both webcam and microphone are properly set up, you can run the project:

```bash
python audio_video_processing/scripts/simplified_laptop_processor.py
```

This should now have access to both your webcam and microphone through WSL! 