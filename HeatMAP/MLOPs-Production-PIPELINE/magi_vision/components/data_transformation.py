import os
import sys
import json
import numpy as np
import cv2

from magi_vision.entity.config_entity import DataTransformationConfig
from magi_vision.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from magi_vision.entity.estimator import SpectralPreprocessor
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.utils.main_utils import (
    get_image_files,
    write_json_file,
    ensure_directory,
)


class DataTransformation:
    """
    Stage 3: Data Transformation
    ----------------------------
    Applies the MAGI Spectral Masking preprocessing pipeline:
      1. CLAHE illumination normalization (LAB L-channel)
      2. Compute spectral proxy channels (ExG, GRVI, L*)
      3. Compute per-channel normalization statistics from training set
      4. Save preprocessing config for deployment
    
    Note: Actual image transformation happens at training time via
    tf.data pipeline. This stage computes and saves the normalization
    statistics that the training pipeline will use.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.ingestion_artifact = data_ingestion_artifact
        self.validation_artifact = data_validation_artifact
        self.config = data_transformation_config

    def _compute_channel_statistics(
        self, image_dir: str, sample_size: int = 500
    ) -> dict:
        """
        Compute mean and std for computed spectral channels (ExG, GRVI, L*)
        from a random sample of training images.
        
        These statistics are used for per-channel normalization during
        training and inference.
        """
        all_images = get_image_files(image_dir)
        
        if len(all_images) > sample_size:
            indices = np.random.choice(len(all_images), sample_size, replace=False)
            sample_images = [all_images[i] for i in indices]
        else:
            sample_images = all_images

        exg_values = []
        grvi_values = []
        l_star_values = []

        preprocessor = SpectralPreprocessor(
            image_size=self.config.image_size,
            clahe_clip=self.config.clahe_clip_limit,
            clahe_grid=self.config.clahe_grid_size,
        )

        logging.info(f"Computing channel stats from {len(sample_images)} images...")

        for img_path in sample_images:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                resized = cv2.resize(
                    img, self.config.image_size, interpolation=cv2.INTER_AREA
                )
                rgb = preprocessor.apply_clahe(resized)
                spectral = preprocessor.compute_spectral_channels(rgb)

                exg_values.append(np.mean(spectral["ExG"]))
                grvi_values.append(np.mean(spectral["GRVI"]))
                l_star_values.append(np.mean(spectral["L_star"]))

            except Exception as e:
                logging.warning(f"Skipping {img_path}: {e}")
                continue

        stats = {
            "ExG": {
                "mean": float(np.mean(exg_values)),
                "std": float(np.std(exg_values)) + 1e-6,
            },
            "GRVI": {
                "mean": float(np.mean(grvi_values)),
                "std": float(np.std(grvi_values)) + 1e-6,
            },
            "L_star": {
                "mean": float(np.mean(l_star_values)),
                "std": float(np.std(l_star_values)) + 1e-6,
            },
        }

        logging.info(
            f"Channel statistics computed: "
            f"ExG(μ={stats['ExG']['mean']:.4f}, σ={stats['ExG']['std']:.4f}), "
            f"GRVI(μ={stats['GRVI']['mean']:.4f}, σ={stats['GRVI']['std']:.4f}), "
            f"L*(μ={stats['L_star']['mean']:.4f}, σ={stats['L_star']['std']:.4f})"
        )

        return stats

    def _save_transform_config(self, stats: dict) -> str:
        """Save the full preprocessing configuration for deployment."""
        config = {
            "image_size": list(self.config.image_size),
            "clahe": {
                "clip_limit": self.config.clahe_clip_limit,
                "grid_size": list(self.config.clahe_grid_size),
            },
            "imagenet_normalization": {
                "mean": self.config.imagenet_mean,
                "std": self.config.imagenet_std,
            },
            "computed_channel_normalization": stats,
            "channels": ["R", "G", "B", "ExG", "GRVI", "L_star"],
            "augmentation": {
                "horizontal_flip": self.config.horizontal_flip,
                "rotation_range": self.config.rotation_range,
                "brightness_range": list(self.config.brightness_range),
                "contrast_range": list(self.config.contrast_range),
                "zoom_range": list(self.config.zoom_range),
                "channel_dropout_rate": self.config.channel_dropout_rate,
                "noise_std": self.config.noise_std,
            },
        }

        config_path = os.path.join(
            self.config.transform_config_dir, "preprocessing_config.json"
        )
        write_json_file(config_path, config, replace=True)
        return config_path

    def _count_samples(self, directory: str) -> int:
        """Count total image samples in a directory."""
        return len(get_image_files(directory))

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Execute data transformation pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 3: DATA TRANSFORMATION started")
        logging.info("=" * 60)

        try:
            if not self.validation_artifact.is_valid:
                raise ValueError(
                    f"Data validation failed: {self.validation_artifact.message}"
                )

            ensure_directory(self.config.transform_config_dir)

            # Step 1: Compute normalization statistics from training data
            logging.info("Computing channel normalization statistics...")
            channel_stats = self._compute_channel_statistics(
                self.ingestion_artifact.train_dir
            )

            # Step 2: Save normalization stats
            write_json_file(
                self.config.normalization_stats_path, channel_stats, replace=True
            )
            logging.info(
                f"Normalization stats saved: {self.config.normalization_stats_path}"
            )

            # Step 3: Save full preprocessing config
            config_path = self._save_transform_config(channel_stats)
            logging.info(f"Preprocessing config saved: {config_path}")

            # Step 4: Count samples
            num_train = self._count_samples(self.ingestion_artifact.train_dir)
            num_val = self._count_samples(self.ingestion_artifact.val_dir)
            num_test = self._count_samples(self.ingestion_artifact.test_dir)

            logging.info(
                f"Sample counts: train={num_train}, val={num_val}, test={num_test}"
            )

            artifact = DataTransformationArtifact(
                normalization_stats_path=self.config.normalization_stats_path,
                transform_config_path=config_path,
                num_train_samples=num_train,
                num_val_samples=num_val,
                num_test_samples=num_test,
            )

            logging.info("Stage 3: DATA TRANSFORMATION completed")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
