#!/bin/bash

# Script to install custom WSL2 kernel with webcam support

KERNEL_DIR=~/wsl-kernel/WSL2-Linux-Kernel
KERNEL_FILE="$KERNEL_DIR/arch/x86/boot/bzImage"
WINDOWS_USER=$(cmd.exe /c echo %USERNAME% 2>/dev/null | tr -d '\r')
WINDOWS_DEST="/mnt/c/Users/$WINDOWS_USER/wsl-kernel/"

# Check if kernel exists
if [ ! -f "$KERNEL_FILE" ]; then
    echo "Error: Kernel file not found at $KERNEL_FILE"
    echo "Please make sure you have compiled the kernel successfully."
    exit 1
fi

# Create destination directory in Windows
echo "Creating destination directory in Windows..."
mkdir -p "$WINDOWS_DEST"

# Copy kernel file to Windows
echo "Copying kernel to Windows..."
cp "$KERNEL_FILE" "$WINDOWS_DEST/bzImage"

if [ $? -ne 0 ]; then
    echo "Error: Failed to copy kernel file to Windows."
    exit 1
fi

echo "Kernel copied successfully to: $WINDOWS_DEST/bzImage"

# Create .wslconfig file
echo "Creating .wslconfig file..."
cat > "/mnt/c/Users/$WINDOWS_USER/.wslconfig" << EOF
[wsl2]
kernel=$WINDOWS_DEST/bzImage
EOF

echo "Custom kernel installation complete."
echo ""
echo "To apply changes, you need to restart WSL. Open a PowerShell window as Administrator and run:"
echo "wsl --shutdown"
echo ""
echo "Then close this terminal and open a new one. After that, run:"
echo "./verify_webcam.py"
echo ""
echo "If everything is set up correctly, your webcam should be detected in WSL." 