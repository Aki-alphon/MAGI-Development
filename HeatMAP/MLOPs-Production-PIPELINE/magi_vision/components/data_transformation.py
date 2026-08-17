import os
import sys
import json
import numpy as np
import cv2
import shutil
import multiprocessing
from functools import partial
import tensorflow as tf

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
    Applies the MAGI Spectral Masking preprocessing pipeline to precompute 
    all 8-channel numpy tensors and save them to disk.
    This eliminates the massive CPU bottleneck of tf.py_function during training.
    
      1. Compute 5-channel spectral stats (ExG, GRVI, VARI, GLI, NGBDI) from train set
      2. Pre-process every image into an 8-channel (224, 224, 8) float32 tensor
      3. Save as .npy files keeping the class directory structure
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
        self, image_dir: str, sample_size: int = 1000
    ) -> dict:
        """
        Compute mean and std for computed spectral channels (ExG, GRVI, VARI, GLI, NGBDI)
        from a random sample of training images. Increased sample_size to 1000 for
        more reliable stats without significantly impacting speed.
        """
        all_images = get_image_files(image_dir)
        
        if len(all_images) > sample_size:
            indices = np.random.choice(len(all_images), sample_size, replace=False)
            sample_images = [all_images[i] for i in indices]
        else:
            sample_images = all_images

        # We need a preprocessor just to get the raw un-normalized values
        preprocessor = SpectralPreprocessor(
            image_size=self.config.image_size,
            clahe_clip=self.config.clahe_clip_limit,
            clahe_grid=self.config.clahe_grid_size,
        )

        logging.info(f"Computing channel stats from {len(sample_images)} images...")

        channel_vals = {
            "ExG": [], "GRVI": [], "VARI": [], "GLI": [], "NGBDI": []
        }

        for img_path in sample_images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                resized = cv2.resize(
                    img, self.config.image_size, interpolation=cv2.INTER_AREA
                )
                rgb = preprocessor.apply_clahe(resized)
                spectral = preprocessor.compute_spectral_channels(rgb)

                for k in channel_vals.keys():
                    channel_vals[k].append(np.mean(spectral[k]))

            except Exception as e:
                logging.warning(f"Skipping {img_path}: {e}")
                continue

        stats = {}
        for k in channel_vals.keys():
            stats[k] = {
                "mean": float(np.mean(channel_vals[k])),
                "std": float(np.std(channel_vals[k])) + 1e-6,
            }

        logging.info(
            f"Channel statistics computed: "
            f"VARI(μ={stats['VARI']['mean']:.4f}, σ={stats['VARI']['std']:.4f}), "
            f"GLI(μ={stats['GLI']['mean']:.4f}, σ={stats['GLI']['std']:.4f})"
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
            "channels": ["R", "G", "B", "ExG", "GRVI", "VARI", "GLI", "NGBDI"],
        }

        config_path = os.path.join(
            self.config.transform_config_dir, "preprocessing_config.json"
        )
        write_json_file(config_path, config, replace=True)
        return config_path

    # ── Parallel worker (module-level for multiprocessing picklability) ────────

    @staticmethod
    def _process_one_image(args: tuple) -> bytes | None:
        """
        Worker function called by multiprocessing.Pool.
        Reads one image, runs spectral preprocessing, serialises to TFRecord bytes.
        Returns None on failure so the main process can skip it.
        """
        img_path, label, image_size, clahe_clip, clahe_grid, computed_stats = args
        try:
            import cv2, numpy as np
            from magi_vision.entity.estimator import SpectralPreprocessor
            preprocessor = SpectralPreprocessor(
                image_size=image_size,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                computed_stats=computed_stats,
            )
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            # float16 halves file size and I/O bandwidth vs float32
            tensor = preprocessor.preprocess(img).astype(np.float16)
            tensor_bytes = tensor.tobytes()

            import tensorflow as tf
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_bytes])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()
        except Exception:
            return None

    def _precompute_tensors(
        self,
        source_dir: str,
        target_dir: str,
        preprocessor: SpectralPreprocessor,
        num_workers: int | None = None,
        num_shards: int = 4,
    ) -> int:
        """
        Walks source_dir, runs spectral preprocessing in parallel (multiprocessing.Pool),
        and writes sharded TFRecord files in target_dir.

        Sharding (num_shards=4) gives the tf.data pipeline multiple files to
        interleave, improving I/O throughput during training by ~20%.
        """
        os.makedirs(target_dir, exist_ok=True)
        class_names = sorted(os.listdir(source_dir))
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        # Collect all (path, label) pairs
        all_tasks = []
        for class_name in class_names:
            class_src = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_src):
                continue
            for img_path in get_image_files(class_src):
                all_tasks.append((
                    str(img_path),
                    class_to_idx[class_name],
                    self.config.image_size,
                    self.config.clahe_clip_limit,
                    self.config.clahe_grid_size,
                    preprocessor.computed_stats,
                ))

        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 4)  # Colab gives 2 vCPUs

        logging.info(
            f"Preprocessing {len(all_tasks)} images with "
            f"{num_workers} workers → {num_shards} TFRecord shards in {target_dir}"
        )

        # Parallel preprocessing
        with multiprocessing.Pool(processes=num_workers) as pool:
            serialised = pool.map(DataTransformation._process_one_image, all_tasks)

        # Drop failures
        serialised = [s for s in serialised if s is not None]
        processed_count = len(serialised)
        skipped = len(all_tasks) - processed_count
        if skipped > 0:
            logging.warning(f"Skipped {skipped} images (read/preprocessing errors)")

        # Shuffle before writing (important: TFRecord order becomes training order)
        import random
        random.shuffle(serialised)

        # Write sharded TFRecords
        shard_size = max(1, len(serialised) // num_shards)
        for shard_idx in range(num_shards):
            shard_path = os.path.join(target_dir, f"dataset_{shard_idx:04d}-of-{num_shards:04d}.tfrecord")
            start = shard_idx * shard_size
            end   = start + shard_size if shard_idx < num_shards - 1 else len(serialised)
            with tf.io.TFRecordWriter(shard_path) as writer:
                for record in serialised[start:end]:
                    writer.write(record)

        logging.info(f"  Written {processed_count} records across {num_shards} shards")
        return processed_count

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Execute data transformation pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 3: DATA TRANSFORMATION started (Pre-computing .npy tensors)")
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

            # Step 4: Precompute tensors to disk
            logging.info("Precomputing 8-channel tensors to .npy files (this takes a few minutes)...")
            preprocessor = SpectralPreprocessor(
                image_size=self.config.image_size,
                clahe_clip=self.config.clahe_clip_limit,
                clahe_grid=self.config.clahe_grid_size,
                computed_stats=channel_stats
            )
            
            num_train = self._precompute_tensors(self.ingestion_artifact.train_dir, self.config.train_npy_dir, preprocessor)
            num_val = self._precompute_tensors(self.ingestion_artifact.val_dir, self.config.val_npy_dir, preprocessor)
            num_test = self._precompute_tensors(self.ingestion_artifact.test_dir, self.config.test_npy_dir, preprocessor)

            logging.info(
                f"Precomputed tensor counts: train={num_train}, val={num_val}, test={num_test}"
            )

            artifact = DataTransformationArtifact(
                normalization_stats_path=self.config.normalization_stats_path,
                transform_config_path=config_path,
                num_train_samples=num_train,
                num_val_samples=num_val,
                num_test_samples=num_test,
                train_npy_dir=self.config.train_npy_dir,
                val_npy_dir=self.config.val_npy_dir,
                test_npy_dir=self.config.test_npy_dir,
            )

            logging.info("Stage 3: DATA TRANSFORMATION completed")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
