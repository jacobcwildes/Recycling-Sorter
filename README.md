## Overview:
This project aims to emulate an automated recycling sorter.

This project uses a Raspberry Pi, a Raspberry Pi camera, microphone, a TFT display,

and a servo motor to identify and output a decision based on audio and visual

inputs.

## Setting up the Raspberry Pi camera:

## NOTE: If running Bullseye, do not follow these steps. The camera is detected by default

1) Enable legacy support for the raspi-cam 
```bash
sudo raspi-config
```

2) Use the arrow keys to navigate to ```Interface Options``` and then hit ```Enter```
Make sure that ```Legacy Camera/Enable/disable legacy camera support``` is enabled

3) Reboot the raspberry pi

Ensure the camera is working: 
```bash
libcamera-jpeg -o Desktop/image.jpeg
```
If using a version other than Bullseye, replace ```libcamera``` with ```raspistill```

## Download requisite packages:

1) Make sure that opencv is installed on the system
```bash
pip3 install opencv-python
```

2) Make sure that PiCamera OpenCV module is installed:
```bash
pip3 install picamera[array]
```
