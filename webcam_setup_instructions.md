# Webcam Setup Instructions for WSL

This guide will help you share your webcam from Windows to WSL using USBIPD.

## Prerequisites
- Windows 11 with WSL2
- USBIPD-Win installed (already verified)
- Linux tools and v4l-utils installed in WSL (already done)

## Steps to Share the Webcam

### 1. Find Your Webcam ID in Windows
1. Open a PowerShell window as Administrator
2. Run the following command to list all USB devices:
   ```
   usbipd list
   ```
3. Look for your webcam device in the list. It should have a name like "Integrated_Webcam_HD" or "USB Camera"
4. Note the busid of your webcam (it will look like `1-4` or `2-3#3#4`)

### 2. Bind the Webcam for Sharing
1. In the same PowerShell window, run:
   ```
   usbipd bind --busid <your-webcam-busid>
   ```
   Replace `<your-webcam-busid>` with the actual busid you found.

### 3. Attach the Webcam to WSL
1. With the PowerShell window still open, run:
   ```
   usbipd attach --wsl --busid <your-webcam-busid>
   ```
   Replace `<your-webcam-busid>` with the same busid.

### 4. Verify the Webcam in WSL
1. In your WSL terminal, run:
   ```
   lsusb
   ```
   You should see your webcam device listed.

2. Run our verification script:
   ```
   ./verify_webcam.py
   ```
   This will check if the webcam is accessible through OpenCV.

### Troubleshooting

If the webcam is not detected:
1. Make sure Windows and WSL are up to date
2. Try unplugging and re-plugging the webcam if it's external
3. Try a different USB port
4. Restart WSL with `wsl --shutdown` in PowerShell, then restart your WSL terminal

### Notes
- The webcam will remain attached until you detach it or restart WSL
- To detach the webcam, run in PowerShell:
  ```
  usbipd detach --busid <your-webcam-busid>
  ```
- You may need to repeat this process each time you restart your computer or WSL

## Running the Audio-Video Processing Project
Once your webcam is properly set up, you can run the project:

```bash
python audio_video_processing/scripts/simplified_laptop_processor.py
```

This should now have access to your webcam and microphone through WSL! 