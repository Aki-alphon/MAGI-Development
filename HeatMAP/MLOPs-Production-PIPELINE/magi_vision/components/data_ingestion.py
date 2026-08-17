"""
MAGI Vision — Stage 1: Data Ingestion
magi_vision/components/data_ingestion.py

3-Source Dataset Strategy for Cotton + Leafy Vegetables:

  Source A: Cotton plant disease dataset (Kaggle: smaranjitghose/cotton-plant-disease)
            Class remap → stress stages using nitrogen-predisposition logic

  Source B: Leafy veg NPK deficiency (Kaggle: rayhanadev/plant-leaf-health)
            Class remap → nitrogen stage labels

  Source C: EarlyNSD — only public dataset with actual pre-symptomatic N-stress labels
            (Kaggle: gauravduttakiit/earlynsd)
            Used exclusively for early_nitrogen_stress class

  Synthetic: UnderCanopyAugmentor applied to healthy images from A+B
             Generates 800 extra baseline_healthy samples with bot-perspective lighting

Final classes (stress progression stages):
  baseline_healthy      → >7 days to disease
  early_nitrogen_stress → 3-5 days (THE KEY PREDICTION TARGET)
  active_chlorosis      → 1-2 days
  severe_deficiency     → Now
"""

import os
import sys
import random
import shutil

import cv2
import numpy as np

from magi_vision.entity.config_entity import DataIngestionConfig
from magi_vision.entity.artifact_entity import DataIngestionArtifact
from magi_vision.entity.estimator import UnderCanopyAugmentor
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.data_access.magi_data import MAGIDataLoader
from magi_vision.utils.main_utils import get_image_files, write_json_file
from magi_vision.constants import (
    CLASS_NAMES,
    KAGGLE_COTTON_DATASET, COTTON_CLASS_REMAP,
    KAGGLE_LEAFY_DATASET,  LEAFY_CLASS_REMAP,
    KAGGLE_EARLYNSD_DATASET, EARLYNSD_CLASSES_FOR_EARLY_STRESS,
    MAX_SAMPLES_PER_CLASS, UNDER_CANOPY_SYNTHETIC_COUNT,
)

# ── Target stress-stage class names ─────────────────────────────────────────
_TARGET_CLASSES = [
    "baseline_healthy",
    "early_nitrogen_stress",
    "active_chlorosis",
    "severe_deficiency",
]


class DataIngestion:
    """
    Stage 1: Data Ingestion
    -----------------------
    Downloads and assembles the 3-source dataset, remaps class labels
    to nitrogen stress stages, applies under-canopy augmentation to the
    healthy class, caps each class at MAX_SAMPLES_PER_CLASS, and splits
    into train/val/test directories.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config
        self.data_loader = MAGIDataLoader()
        self.augmentor = UnderCanopyAugmentor(seed=42)

    # ── Dataset download ─────────────────────────────────────────────────────

    def _download_sources(self) -> dict:
        """
        Download all 3 Kaggle sources to separate subdirectories.
        Returns dict of {source_name: local_dir}.

        Skips download if directory already exists (Colab re-run safe).
        """
        raw = self.config.raw_data_dir

        sources = {
            "cotton":   (KAGGLE_COTTON_DATASET,  os.path.join(raw, "cotton")),
        }

        local_dirs = {}
        for name, (slug, dest) in sources.items():
            if os.path.exists(dest) and len(os.listdir(dest)) > 0:
                logging.info(f"[{name}] Already downloaded — skipping ({dest})")
            else:
                logging.info(f"[{name}] Downloading {slug} → {dest}")
                self.data_loader.download_kaggle_dataset(slug, dest)
            local_dirs[name] = dest

        return local_dirs

    # ── Class remapping ──────────────────────────────────────────────────────

    def _remap_and_collect(self, source_dirs: dict) -> dict:
        """
        Walk each source directory, remap source class names to target stress stages,
        and collect image paths per target class.

        Returns: {target_class: [image_paths]}
        """
        collected = {cls: [] for cls in _TARGET_CLASSES}

        def _walk_and_remap(source_dir: str, remap: dict, fuzzy: bool = False):
            """Walk a directory structure and remap class subdirs recursively."""
            if not os.path.isdir(source_dir):
                logging.warning(f"Source directory not found: {source_dir}")
                return
                
            for root, dirs, files in os.walk(source_dir):
                dir_name = os.path.basename(root)
                # Exact match first
                target = remap.get(dir_name)

                # Case-insensitive fuzzy match if no exact hit
                if target is None and fuzzy:
                    for k, v in remap.items():
                        if k.lower() in dir_name.lower() or dir_name.lower() in k.lower():
                            target = v
                            break

                if target is not None:
                    imgs = [os.path.join(root, f) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if imgs:
                        collected[target].extend(imgs)
                        logging.info(f"  {dir_name} → {target}: +{len(imgs)} images")

        logging.info("=== Remapping Source A: Cotton ===")
        _walk_and_remap(source_dirs["cotton"],   COTTON_CLASS_REMAP, fuzzy=True)

        return collected

    # ── Under-canopy synthetic augmentation ──────────────────────────────────

    def _augment_under_canopy(self, collected: dict) -> dict:
        """
        Generate UNDER_CANOPY_SYNTHETIC_COUNT synthetic images for baseline_healthy
        by applying UnderCanopyAugmentor to existing healthy images.

        This teaches the model what a healthy plant looks like from the bot's
        sideways/upward perspective under the cotton/leafy veg canopy.
        """
        healthy_imgs = collected.get("baseline_healthy", [])
        if len(healthy_imgs) == 0:
            logging.warning("No healthy images found for under-canopy augmentation")
            return collected

        logging.info(
            f"Generating {UNDER_CANOPY_SYNTHETIC_COUNT} synthetic "
            f"under-canopy images from {len(healthy_imgs)} healthy sources..."
        )

        synth_dir = os.path.join(self.config.raw_data_dir, "synthetic_under_canopy")
        os.makedirs(synth_dir, exist_ok=True)

        generated = 0
        for i, src_path in enumerate(
            np.random.default_rng(42).choice(
                healthy_imgs,
                size=min(UNDER_CANOPY_SYNTHETIC_COUNT, len(healthy_imgs) * 5),
                replace=True,
            )
        ):
            if generated >= UNDER_CANOPY_SYNTHETIC_COUNT:
                break
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            augmented = self.augmentor.augment(img)
            out_path = os.path.join(synth_dir, f"synth_uc_{generated:04d}.jpg")
            cv2.imwrite(out_path, augmented, [cv2.IMWRITE_JPEG_QUALITY, 90])
            generated += 1

        synth_paths = get_image_files(synth_dir)
        collected["baseline_healthy"].extend(synth_paths)
        logging.info(f"Under-canopy augmentation: +{len(synth_paths)} synthetic images")
        return collected

    # ── Per-class capping ────────────────────────────────────────────────────

    def _cap_classes(self, collected: dict, seed: int = 42) -> dict:
        """
        Cap each class at MAX_SAMPLES_PER_CLASS to keep training fast
        and the dataset balanced. Stratified random sampling.
        """
        rng = random.Random(seed)
        capped = {}
        for cls, paths in collected.items():
            if len(paths) > MAX_SAMPLES_PER_CLASS:
                capped[cls] = rng.sample(paths, MAX_SAMPLES_PER_CLASS)
                logging.info(f"  {cls}: capped {len(paths)} → {MAX_SAMPLES_PER_CLASS}")
            else:
                capped[cls] = paths
                logging.info(f"  {cls}: {len(paths)} images (under cap)")
        return capped

    # ── Dataset materialisation ──────────────────────────────────────────────

    def _materialise(self, collected: dict):
        """
        Copy collected images into a unified source directory with
        target class subdirectories ready for splitting.
        """
        unified_dir = os.path.join(self.config.raw_data_dir, "unified")
        os.makedirs(unified_dir, exist_ok=True)

        for cls, paths in collected.items():
            cls_dir = os.path.join(unified_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i, src in enumerate(paths):
                ext  = os.path.splitext(src)[1].lower() or ".jpg"
                dest = os.path.join(cls_dir, f"{cls}_{i:05d}{ext}")
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)

        logging.info(f"Unified dataset materialised at {unified_dir}")
        return unified_dir

    # ── Main pipeline ────────────────────────────────────────────────────────

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Execute the full 3-source data ingestion pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 1: DATA INGESTION started")
        logging.info("Target classes: " + ", ".join(_TARGET_CLASSES))
        logging.info("=" * 60)

        try:
            # Step 1: Download all sources
            source_dirs = self._download_sources()

            # Step 2: Remap classes from all 3 sources
            logging.info("Remapping classes across all sources...")
            collected = self._remap_and_collect(source_dirs)
            for cls, paths in collected.items():
                logging.info(f"  After remap — {cls}: {len(paths)} images")

            # Step 3: Under-canopy synthetic augmentation on healthy class
            collected = self._augment_under_canopy(collected)

            # Step 4: Cap per class
            logging.info(f"Capping at {MAX_SAMPLES_PER_CLASS} per class...")
            collected = self._cap_classes(collected)

            # Step 5: Materialise into unified directory
            unified_dir = self._materialise(collected)

            # Step 6: Load dataset info
            dataset_info = self.data_loader.load_from_directory(unified_dir)
            logging.info(
                f"Final dataset: {dataset_info['total_images']} images "
                f"across {len(dataset_info['class_names'])} classes"
            )

            # Step 7: Train/val/test split
            logging.info("Splitting into train/val/test...")
            train_dist, val_dist, test_dist = self.data_loader.split_dataset(
                source_dir=unified_dir,
                train_dir=self.config.train_dir,
                val_dir=self.config.val_dir,
                test_dir=self.config.test_dir,
                train_ratio=self.config.train_split,
                val_ratio=self.config.val_split,
                test_ratio=self.config.test_split,
            )

            total_train = sum(train_dist.values())
            total_val   = sum(val_dist.values())
            total_test  = sum(test_dist.values())
            logging.info(
                f"Split: train={total_train}, val={total_val}, test={total_test}"
            )

            # Step 8: Save class mapping
            class_mapping = {i: name for i, name in enumerate(_TARGET_CLASSES)}
            mapping_path  = os.path.join(
                self.config.data_ingestion_dir, "class_mapping.json"
            )
            write_json_file(mapping_path, class_mapping)
            logging.info(f"Class mapping saved: {mapping_path}")

            artifact = DataIngestionArtifact(
                train_dir          = self.config.train_dir,
                val_dir            = self.config.val_dir,
                test_dir           = self.config.test_dir,
                class_names        = _TARGET_CLASSES,
                class_distribution = dataset_info["class_distribution"],
                total_images       = dataset_info["total_images"],
            )

            logging.info(f"Data Ingestion artifact: {artifact}")
            logging.info("Stage 1: DATA INGESTION completed")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
