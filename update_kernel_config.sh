#!/bin/bash

# Script to update WSL2 kernel configuration for webcam support

KERNEL_DIR=~/wsl-kernel/WSL2-Linux-Kernel

cd $KERNEL_DIR || { echo "Kernel directory not found"; exit 1; }

# Disable BTF generation which was causing the error
sed -i 's/CONFIG_DEBUG_INFO_BTF=y/# CONFIG_DEBUG_INFO_BTF is not set/' .config

# Enable USB Video Class support
echo "CONFIG_USB_VIDEO_CLASS=m" >> .config
echo "CONFIG_USB_VIDEO_CLASS_INPUT_EVDEV=y" >> .config

# Enable Media support explicitly if not already enabled
if ! grep -q "CONFIG_MEDIA_SUPPORT=" .config; then
    echo "CONFIG_MEDIA_SUPPORT=m" >> .config
fi

# Make sure V4L2 core is enabled
if ! grep -q "CONFIG_VIDEO_V4L2=m" .config; then
    echo "CONFIG_VIDEO_V4L2=m" >> .config
fi

# Enable Media USB adapters
echo "CONFIG_MEDIA_USB_SUPPORT=y" >> .config

# Enable Realtek driver (based on the webcam hardware we detected)
echo "CONFIG_VIDEO_REALTEK=m" >> .config

echo "Kernel configuration updated for webcam support."
echo "Next step: Run 'cd $KERNEL_DIR && make -j4' to compile the kernel" 