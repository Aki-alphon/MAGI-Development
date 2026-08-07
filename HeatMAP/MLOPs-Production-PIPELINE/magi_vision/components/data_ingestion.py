import os
import sys

from magi_vision.entity.config_entity import DataIngestionConfig
from magi_vision.entity.artifact_entity import DataIngestionArtifact
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.data_access.magi_data import MAGIDataLoader
from magi_vision.utils.main_utils import get_class_distribution, write_json_file


class DataIngestion:
    """
    Stage 1: Data Ingestion
    -----------------------
    Downloads or locates the image dataset, verifies structure,
    and splits into train/val/test directories.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config
        self.data_loader = MAGIDataLoader()

    def _get_dataset_source(self) -> str:
        """
        Determine dataset source. Checks for:
          1. Local 'data/' directory in project root
          2. Environment variable MAGI_DATASET_PATH
          3. Kaggle download
        """
        # Check local data directory
        local_data = os.path.join(os.getcwd(), "data")
        if os.path.exists(local_data):
            # Look for class subdirectories
            subdirs = [
                d for d in os.listdir(local_data)
                if os.path.isdir(os.path.join(local_data, d))
            ]
            if len(subdirs) > 0:
                logging.info(f"Using local dataset: {local_data}")
                return local_data

        # Check environment variable
        env_path = os.environ.get("MAGI_DATASET_PATH")
        if env_path and os.path.exists(env_path):
            logging.info(f"Using dataset from env: {env_path}")
            return env_path

        # Fall back to Kaggle download
        logging.info("No local dataset found. Downloading from Kaggle...")
        kaggle_dataset = os.environ.get(
            "MAGI_KAGGLE_DATASET", "vipoooool/new-plant-diseases-dataset"
        )
        download_path = self.config.raw_data_dir
        self.data_loader.download_kaggle_dataset(kaggle_dataset, download_path)

        # Navigate into the extracted directory
        # PlantVillage structure: raw_data/New Plant Diseases Dataset(Augmented)/...
        for root, dirs, _ in os.walk(download_path):
            for d in dirs:
                sub_path = os.path.join(root, d)
                sub_items = os.listdir(sub_path)
                # Check if this dir contains class subdirectories
                class_dirs = [
                    s for s in sub_items
                    if os.path.isdir(os.path.join(sub_path, s))
                ]
                if len(class_dirs) >= 2:
                    return sub_path

        return download_path

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Execute data ingestion pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 1: DATA INGESTION started")
        logging.info("=" * 60)

        try:
            # Step 1: Get dataset source
            source_dir = self._get_dataset_source()
            dataset_info = self.data_loader.load_from_directory(source_dir)

            logging.info(
                f"Found {dataset_info['total_images']} images "
                f"across {len(dataset_info['class_names'])} classes"
            )

            # Step 2: Split into train/val/test
            logging.info("Splitting dataset into train/val/test...")
            train_dist, val_dist, test_dist = self.data_loader.split_dataset(
                source_dir=source_dir,
                train_dir=self.config.train_dir,
                val_dir=self.config.val_dir,
                test_dir=self.config.test_dir,
                train_ratio=self.config.train_split,
                val_ratio=self.config.val_split,
                test_ratio=self.config.test_split,
            )

            total_train = sum(train_dist.values())
            total_val = sum(val_dist.values())
            total_test = sum(test_dist.values())

            logging.info(
                f"Split complete: train={total_train}, "
                f"val={total_val}, test={total_test}"
            )

            # Step 3: Save class mapping
            class_mapping = {
                i: name for i, name in enumerate(dataset_info["class_names"])
            }
            mapping_path = os.path.join(
                self.config.data_ingestion_dir, "class_mapping.json"
            )
            write_json_file(mapping_path, class_mapping)

            artifact = DataIngestionArtifact(
                train_dir=self.config.train_dir,
                val_dir=self.config.val_dir,
                test_dir=self.config.test_dir,
                class_names=dataset_info["class_names"],
                class_distribution=dataset_info["class_distribution"],
                total_images=dataset_info["total_images"],
            )

            logging.info(f"Data Ingestion artifact: {artifact}")
            logging.info("Stage 1: DATA INGESTION completed")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
