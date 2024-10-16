# Image Encryption and Decryption Application

## Overview

This Python application provides functionalities to encrypt and decrypt images using various techniques, including chaotic sequences, scrambling, and bitwise operations. The application utilizes OpenCV and NumPy libraries for image processing and supports common image formats such as JPG, PNG, and BMP.

## Features

- **Image Loading:** Load images from a specified folder.
- **Encryption:**
  - Convert images to grayscale and binary format.
  - Use chaotic sequences for scrambling.
  - Apply a Rubik's Cube-like scrambling technique for additional security.
- **Decryption:** Reverse the encryption process to retrieve the original image.
- **Batch Processing:** Encrypt and decrypt all images in a specified folder.

## Requirements

- Python 3.x
- OpenCV
- NumPy

You can install the required libraries using the following command:

```bash
pip install opencv-python numpy
