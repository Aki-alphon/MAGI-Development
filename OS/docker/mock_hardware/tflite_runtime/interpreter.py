"""
MAGI OS — Mock tflite_runtime
docker/mock_hardware/tflite_runtime/interpreter.py

Returns random but correctly-shaped output tensors so all
postprocessing code runs end-to-end without a real model file.
"""

import numpy as np
import random


def load_delegate(path: str, options: dict = None):
    return None   # XNNPACK not available in mock — falls back gracefully


class Interpreter:
    """
    Simulates TFLite interpreter.
    Automatically infers output shape from the model filename.
    """

    # Map model name keywords → output spec
    _OUTPUT_SPECS = {
        "celebi": {
            # Plant health classifier: 4 classes
            # [healthy, mild_stress, moderate_stress, severe_stress]
            "outputs": [
                np.array([[0.7, 0.2, 0.08, 0.02]], dtype=np.float32),  # softmax probs
            ]
        },
        "gengar": {
            # Classification: 10 scene classes
            "outputs": [np.random.dirichlet(np.ones(10)).reshape(1, 10).astype(np.float32)]
        },
        "lugia": {
            # Fusion: 5 action logits
            "outputs": [np.random.rand(1, 5).astype(np.float32)]
        },
    }

    def __init__(self, model_path: str = "", experimental_delegates=None, num_threads: int = 1):
        self._model_path = model_path
        self._tensors    = {}
        self._key        = "lugia"  # default

        for k in self._OUTPUT_SPECS:
            if k in model_path.lower():
                self._key = k
                break

        print(f"[mock_tflite] Interpreter stub for '{self._key}' ({model_path})")

    def allocate_tensors(self):
        pass

    def get_input_details(self) -> list:
        # Celebi expects (1, 224, 224, 6) — 6-channel spectral tensor
        if self._key == "celebi":
            return [{"index": 0, "shape": [1, 224, 224, 6], "dtype": np.float32}]
        return [{"index": 0, "shape": [1, 224, 224, 3], "dtype": np.float32}]

    def get_output_details(self) -> list:
        spec = self._OUTPUT_SPECS.get(self._key, {"outputs": [np.zeros((1, 5), dtype=np.float32)]})
        return [{"index": i} for i in range(len(spec["outputs"]))]

    def set_tensor(self, index: int, data: np.ndarray):
        self._tensors[index] = data

    def invoke(self):
        """Simulate random inference with small delay."""
        import time
        time.sleep(random.uniform(0.010, 0.050))   # 10–50 ms mock inference
        # Refresh random output each invoke
        spec = self._OUTPUT_SPECS.get(self._key, {"outputs": [np.zeros((1, 5), dtype=np.float32)]})
        if self._key == "celebi":
            # Simulate a realistic plant health distribution:
            # mostly healthy, occasional mild/moderate stress patches
            probs = np.random.dirichlet(np.array([6.0, 2.5, 1.0, 0.5]))
            spec["outputs"][0] = probs.reshape(1, 4).astype(np.float32)
        elif self._key == "gengar":
            spec["outputs"][0] = np.random.dirichlet(np.ones(10)).reshape(1, 10).astype(np.float32)

    def get_tensor(self, index: int) -> np.ndarray:
        spec = self._OUTPUT_SPECS.get(self._key, {"outputs": [np.zeros((1, 5), dtype=np.float32)]})
        if index < len(spec["outputs"]):
            return spec["outputs"][index]
        return np.zeros((1, 1), dtype=np.float32)
