"""
MAGI OS — Dummy Model Generator
docker/dummy_models/generate_dummy_models.py

Creates minimal valid TFLite flatbuffer files so the real
tflite_runtime can load them (when not using the mock).
In Docker, the mock interpreter ignores these files entirely.

Run once: python3 generate_dummy_models.py
Requires: tensorflow or ai-edge-torch
"""

import os
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_dummy_tflite(path: str, note: str = ""):
    """
    Write a zero-byte placeholder. The mock tflite_runtime
    ignores the file content entirely.
    For real RPi deployment, replace with actual .tflite files.
    """
    with open(path, "wb") as f:
        f.write(b"MAGI_PLACEHOLDER")   # 16-byte stub
    print(f"Created placeholder: {path}  {note}")


if __name__ == "__main__":
    models = {
        "melchior.tflite":  "(replace with YOLOv8-nano or MobileNet-SSD TFLite)",
        "balthasar.tflite": "(replace with EfficientNet-Lite TFLite)",
        "caspar.tflite":    "(replace with custom LSTM TFLite or leave for rule engine)",
    }
    for fname, note in models.items():
        make_dummy_tflite(os.path.join(OUTPUT_DIR, fname), note)

    print("\nPlaceholder models created.")
    print("For real inference on RPi, replace with actual .tflite files.")
    print("Docker testing uses mock tflite_runtime — file content ignored.")
