import os
import sys
import json
import numpy as np
import yaml

from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MAGIVisionException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file, default_flow_style=False)
    except Exception as e:
        raise MAGIVisionException(e, sys) from e


def read_json_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise MAGIVisionException(e, sys) from e


def write_json_file(file_path: str, content: dict, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(content, f, indent=2)
    except Exception as e:
        raise MAGIVisionException(e, sys) from e


def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """Save numpy array data to file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, array)
    except Exception as e:
        raise MAGIVisionException(e, sys) from e


def load_numpy_array(file_path: str) -> np.ndarray:
    """Load numpy array data from file."""
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise MAGIVisionException(e, sys) from e


def get_image_files(directory: str, extensions: list = None) -> list:
    """Recursively find all image files in a directory."""
    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "bmp"]
    image_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().split(".")[-1] in extensions:
                image_files.append(os.path.join(root, f))
    return image_files


def get_class_distribution(directory: str) -> dict:
    """Get the number of images per class from a directory with class subfolders."""
    distribution = {}
    if not os.path.exists(directory):
        return distribution
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            count = len(get_image_files(class_dir))
            distribution[class_name] = count
    return distribution


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
