import os
import sys
import shutil
import logging
from typing import Tuple

from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.utils.main_utils import get_image_files, get_class_distribution


class MAGIDataLoader:
    """
    Handles loading and organizing image datasets for MAGI vision training.
    Replaces MongoDB data access with filesystem-based image directory loading.

    Supports:
      - Local directory datasets (PlantVillage, PlantDoc, custom)
      - Kaggle dataset download via kaggle CLI
      - Stratified train/val/test splitting
    """

    def __init__(self):
        logging.info("MAGIDataLoader initialized")

    def load_from_directory(self, source_dir: str) -> dict:
        """
        Load dataset info from a directory with class subfolders.

        Expected structure:
            source_dir/
                class_a/
                    img001.jpg
                    img002.jpg
                class_b/
                    img003.jpg
                ...

        Returns:
            dict with class_names, class_distribution, total_images, source_dir
        """
        try:
            if not os.path.exists(source_dir):
                raise FileNotFoundError(f"Dataset directory not found: {source_dir}")

            class_names = sorted([
                d for d in os.listdir(source_dir)
                if os.path.isdir(os.path.join(source_dir, d))
            ])

            if len(class_names) == 0:
                raise ValueError(f"No class subdirectories found in {source_dir}")

            distribution = get_class_distribution(source_dir)
            total = sum(distribution.values())

            logging.info(
                f"Dataset loaded: {len(class_names)} classes, "
                f"{total} total images from {source_dir}"
            )

            return {
                "class_names": class_names,
                "class_distribution": distribution,
                "total_images": total,
                "source_dir": source_dir,
            }

        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def split_dataset(
        self,
        source_dir: str,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[dict, dict, dict]:
        """
        Stratified split of image dataset into train/val/test directories.
        Copies files to preserve the original dataset.
        """
        import random

        try:
            random.seed(seed)

            for d in [train_dir, val_dir, test_dir]:
                os.makedirs(d, exist_ok=True)

            class_names = sorted([
                d for d in os.listdir(source_dir)
                if os.path.isdir(os.path.join(source_dir, d))
            ])

            train_dist, val_dist, test_dist = {}, {}, {}

            for class_name in class_names:
                class_src = os.path.join(source_dir, class_name)
                images = get_image_files(class_src)
                random.shuffle(images)

                n = len(images)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)

                train_imgs = images[:n_train]
                val_imgs = images[n_train: n_train + n_val]
                test_imgs = images[n_train + n_val:]

                # Copy to split directories
                for split_imgs, split_dir in [
                    (train_imgs, train_dir),
                    (val_imgs, val_dir),
                    (test_imgs, test_dir),
                ]:
                    class_dest = os.path.join(split_dir, class_name)
                    os.makedirs(class_dest, exist_ok=True)
                    for img_path in split_imgs:
                        fname = os.path.basename(img_path)
                        shutil.move(img_path, os.path.join(class_dest, fname))

                train_dist[class_name] = len(train_imgs)
                val_dist[class_name] = len(val_imgs)
                test_dist[class_name] = len(test_imgs)

                logging.info(
                    f"  {class_name}: train={len(train_imgs)}, "
                    f"val={len(val_imgs)}, test={len(test_imgs)}"
                )

            return train_dist, val_dist, test_dist

        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    @staticmethod
    def download_kaggle_dataset(
        dataset_name: str, destination: str
    ) -> str:
        """
        Download a dataset from Kaggle.

        Args:
            dataset_name: e.g. 'vipoooool/new-plant-diseases-dataset'
            destination: local path to download to
        Returns:
            path to extracted dataset directory
        """
        try:
            os.makedirs(destination, exist_ok=True)
            os.system(
                f"kaggle datasets download -d {dataset_name} "
                f"-p {destination} --unzip && rm -f {destination}/*.zip"
            )
            logging.info(f"Kaggle dataset '{dataset_name}' downloaded to {destination}")
            return destination
        except Exception as e:
            raise MAGIVisionException(e, sys) from e
