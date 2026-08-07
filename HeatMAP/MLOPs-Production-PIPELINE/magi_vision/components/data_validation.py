import os
import sys
from PIL import Image

from magi_vision.entity.config_entity import DataValidationConfig
from magi_vision.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.utils.main_utils import (
    get_image_files,
    get_class_distribution,
    write_yaml_file,
)


class DataValidation:
    """
    Stage 2: Data Validation
    ------------------------
    Validates the ingested image dataset:
      - Image file integrity (not corrupt/truncated)
      - Supported formats
      - Minimum image dimensions
      - Class balance checks
      - Minimum samples per class
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        self.ingestion_artifact = data_ingestion_artifact
        self.config = data_validation_config

    def _validate_image(self, image_path: str) -> bool:
        """Check if an image file is valid (not corrupt/truncated)."""
        try:
            ext = image_path.lower().split(".")[-1]
            if ext not in self.config.supported_formats:
                return False

            with Image.open(image_path) as img:
                img.verify()  # Check for corruption

            # Re-open to check dimensions (verify() closes the file)
            with Image.open(image_path) as img:
                w, h = img.size
                if w < self.config.min_image_size[0] or h < self.config.min_image_size[1]:
                    return False

            return True

        except Exception:
            return False

    def _validate_split(self, split_dir: str) -> tuple:
        """Validate all images in a split directory. Returns (valid_count, corrupt_list)."""
        corrupt_images = []
        valid_count = 0
        all_images = get_image_files(split_dir)

        for img_path in all_images:
            if self._validate_image(img_path):
                valid_count += 1
            else:
                corrupt_images.append(img_path)
                logging.warning(f"Invalid/corrupt image: {img_path}")

        return valid_count, corrupt_images

    def _check_class_balance(self, distribution: dict) -> tuple:
        """Check if class distribution is severely imbalanced."""
        if not distribution:
            return False, "No classes found"

        counts = list(distribution.values())
        max_count = max(counts)
        min_count = min(counts)

        if min_count == 0:
            empty_classes = [k for k, v in distribution.items() if v == 0]
            return False, f"Empty classes found: {empty_classes}"

        ratio = max_count / min_count
        if ratio > self.config.max_imbalance_ratio:
            return (
                False,
                f"Severe class imbalance: ratio {ratio:.1f}:1 "
                f"(max allowed: {self.config.max_imbalance_ratio}:1)",
            )

        return True, f"Class balance OK (ratio {ratio:.1f}:1)"

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Execute data validation pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 2: DATA VALIDATION started")
        logging.info("=" * 60)

        try:
            messages = []
            all_corrupt = []
            is_valid = True

            # Step 1: Validate images in each split
            for split_name, split_dir in [
                ("train", self.ingestion_artifact.train_dir),
                ("val", self.ingestion_artifact.val_dir),
                ("test", self.ingestion_artifact.test_dir),
            ]:
                valid_count, corrupt = self._validate_split(split_dir)
                all_corrupt.extend(corrupt)
                msg = (
                    f"{split_name}: {valid_count} valid images, "
                    f"{len(corrupt)} corrupt"
                )
                messages.append(msg)
                logging.info(msg)

                if len(corrupt) > 0:
                    # Remove corrupt images
                    for img_path in corrupt:
                        os.remove(img_path)
                        logging.info(f"  Removed corrupt: {img_path}")

            # Step 2: Check class distribution
            train_dist = get_class_distribution(self.ingestion_artifact.train_dir)
            balance_ok, balance_msg = self._check_class_balance(train_dist)
            messages.append(balance_msg)
            logging.info(balance_msg)

            if not balance_ok:
                logging.warning(f"Class balance warning: {balance_msg}")
                # Don't fail — just warn (we use class weights in training)

            # Step 3: Check minimum samples per class
            for class_name, count in train_dist.items():
                if count < self.config.min_samples_per_class:
                    msg = (
                        f"WARNING: '{class_name}' has only {count} training samples "
                        f"(min: {self.config.min_samples_per_class})"
                    )
                    messages.append(msg)
                    logging.warning(msg)

            # Step 4: Write validation report
            report = {
                "validation_status": is_valid,
                "messages": messages,
                "corrupt_images_count": len(all_corrupt),
                "class_distribution": train_dist,
                "total_classes": len(train_dist),
            }
            write_yaml_file(self.config.validation_report_path, report, replace=True)

            artifact = DataValidationArtifact(
                is_valid=is_valid,
                message=" | ".join(messages),
                validation_report_path=self.config.validation_report_path,
                corrupt_images=all_corrupt,
                class_distribution=train_dist,
            )

            logging.info(f"Data Validation: {'PASSED' if is_valid else 'FAILED'}")
            logging.info("Stage 2: DATA VALIDATION completed")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
