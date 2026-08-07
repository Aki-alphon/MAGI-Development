import os
import sys
import shutil

from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging


class LocalModelStorage:
    """Local filesystem-based model storage. Replaces AWS S3."""

    def __init__(self, base_path: str = "model_registry"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save_model(self, source_path: str, model_name: str, version: str) -> str:
        try:
            dest_dir = os.path.join(self.base_path, model_name, version)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(source_path))
            shutil.copy2(source_path, dest_path)
            latest_link = os.path.join(self.base_path, model_name, "latest")
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            os.symlink(dest_dir, latest_link)
            logging.info(f"Model saved: {dest_path} (latest -> {version})")
            return dest_path
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def load_model_path(self, model_name: str, version: str = "latest") -> str:
        model_dir = os.path.join(self.base_path, model_name, version)
        if os.path.islink(model_dir):
            model_dir = os.path.realpath(model_dir)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model not found: {model_name}/{version}")
        return model_dir

    def list_versions(self, model_name: str) -> list:
        model_base = os.path.join(self.base_path, model_name)
        if not os.path.exists(model_base):
            return []
        return sorted([
            d for d in os.listdir(model_base)
            if os.path.isdir(os.path.join(model_base, d)) and d != "latest"
        ])
