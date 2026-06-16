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
        "melchior": {
            # Detection model: boxes, classes, scores, count
            "outputs": [
                np.zeros((1, 10, 4),  dtype=np.float32),   # boxes
                np.zeros((1, 10),     dtype=np.float32),   # classes
                np.random.rand(1, 10).astype(np.float32),  # scores
                np.array([[3.0]],     dtype=np.float32),   # count
            ]
        },
        "balthasar": {
            # Classification: 10 scene classes
            "outputs": [np.random.dirichlet(np.ones(10)).reshape(1, 10).astype(np.float32)]
        },
        "caspar": {
            # Fusion: 5 action logits
            "outputs": [np.random.rand(1, 5).astype(np.float32)]
        },
    }

    def __init__(self, model_path: str = "", experimental_delegates=None, num_threads: int = 1):
        self._model_path = model_path
        self._tensors    = {}
        self._key        = "caspar"  # default

        for k in self._OUTPUT_SPECS:
            if k in model_path.lower():
                self._key = k
                break

        print(f"[mock_tflite] Interpreter stub for '{self._key}' ({model_path})")

    def allocate_tensors(self):
        pass

    def get_input_details(self) -> list:
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
        if self._key == "melchior":
            scores = np.random.rand(1, 10).astype(np.float32)
            scores[0, :3] = np.random.uniform(0.4, 0.95, 3)  # 3 confident detections
            spec["outputs"][2] = scores
            spec["outputs"][1] = np.array([[0, 14, 2]], dtype=np.float32).reshape(1, 10)  # person, cat, car
        elif self._key == "balthasar":
            spec["outputs"][0] = np.random.dirichlet(np.ones(10)).reshape(1, 10).astype(np.float32)

    def get_tensor(self, index: int) -> np.ndarray:
        spec = self._OUTPUT_SPECS.get(self._key, {"outputs": [np.zeros((1, 5), dtype=np.float32)]})
        if index < len(spec["outputs"]):
            return spec["outputs"][index]
        return np.zeros((1, 1), dtype=np.float32)
