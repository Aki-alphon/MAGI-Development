# MAGI AIML Models

This directory contains the trained artificial intelligence and machine learning weights for the MAGI vision subsystem.

## Contents
- `celebi.tflite`: The main Disease Classification Core model (MobileNetV2 based), heavily quantized and optimized for edge CPU inference (Raspberry Pi 4B).
- `gengar.tflite` / `lugia.tflite`: Other lightweight gating and decision-engine models used by the system.

> **Note**: Raw `.keras` and `.h5` files are stored locally but ignored in version control due to GitHub's file size limits. Only the optimized `.tflite` deployments are tracked.
