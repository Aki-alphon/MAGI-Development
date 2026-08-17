import os
import sys
import shutil
import json
from datetime import datetime

from magi_vision.entity.config_entity import ModelPusherConfig, ModelTrainerConfig
from magi_vision.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    ModelPusherArtifact,
)
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.utils.main_utils import write_json_file, ensure_directory


class ModelPusher:
    """
    Stage 6: Model Pusher
    ---------------------
    Deploys the accepted model:
      1. Copy TFLite model to export directory
      2. Copy preprocessing config and class mapping
      3. Generate deployment manifest
      4. Optionally copy to Pi deployment path
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.trainer_artifact = model_trainer_artifact
        self.evaluation_artifact = model_evaluation_artifact
        self.config = model_pusher_config

    def _generate_manifest(self, version: str, dest_dir: str) -> str:
        """Generate deployment manifest with model metadata."""
        manifest = {
            "model_name": "celebi",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "architecture": "MobileNetV2_8ch",
            "input_shape": [224, 224, 8],
            "input_channels": ["R", "G", "B", "ExG", "GRVI", "VARI", "GLI", "NGBDI"],
            "num_classes": 4,
            "class_names": [
                "baseline_healthy",
                "early_nitrogen_stress",
                "active_chlorosis",
                "severe_deficiency",
            ],
            "health_scores": [1.0, 0.68, 0.35, 0.05],
            "quantization": "float16",
            "metrics": {
                "keras_accuracy": self.evaluation_artifact.keras_accuracy,
                "tflite_accuracy": self.evaluation_artifact.tflite_accuracy,
                "inference_latency_ms": self.evaluation_artifact.tflite_latency_ms,
                "model_size_mb": self.evaluation_artifact.model_size_mb,
                "f1_score": self.trainer_artifact.metric_artifact.f1_score,
            },
            "deployment": {
                "runtime": "tflite",
                "delegate": "xnnpack",
                "num_threads": 4,
                "target_device": "raspberry_pi_4b",
            },
            "files": {
                "tflite_model": "celebi.tflite",
                "class_mapping": "class_mapping.json",
                "normalization_stats": "normalization_stats.json",
                "deployment_manifest": "deployment_manifest.json",
            },
        }

        manifest_path = os.path.join(dest_dir, "deployment_manifest.json")
        write_json_file(manifest_path, manifest, replace=True)
        return manifest_path

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """Execute model pusher pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 6: MODEL PUSHER started")
        logging.info("=" * 60)

        try:
            if not self.evaluation_artifact.is_model_accepted:
                logging.warning("Model was NOT accepted. Skipping deployment.")
                return ModelPusherArtifact(
                    deployed_model_path="",
                    manifest_path="",
                    model_version="rejected",
                )

            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = self.config.tflite_export_dir
            ensure_directory(export_dir)

            # Step 1: Copy TFLite model
            tflite_dest = os.path.join(export_dir, "celebi.tflite")
            shutil.copy2(self.trainer_artifact.tflite_model_path, tflite_dest)
            logging.info(f"TFLite model → {tflite_dest}")

            # Step 2: Copy class mapping
            mapping_dest = os.path.join(export_dir, "class_mapping.json")
            shutil.copy2(self.trainer_artifact.class_mapping_path, mapping_dest)
            logging.info(f"Class mapping → {mapping_dest}")

            # Step 3: Copy preprocessing config if exists
            from magi_vision.constants import DATA_TRANSFORMATION_DIR_NAME
            transform_dir = os.path.join(
                os.path.dirname(os.path.dirname(self.trainer_artifact.trained_model_path)),
                DATA_TRANSFORMATION_DIR_NAME,
            )

            for config_file in ["preprocessing_config.json", "normalization_stats.json"]:
                # Search for config files in the artifact directory
                for root, _, files in os.walk(
                    os.path.dirname(os.path.dirname(self.trainer_artifact.trained_model_path))
                ):
                    if config_file in files:
                        src = os.path.join(root, config_file)
                        dst = os.path.join(export_dir, config_file)
                        shutil.copy2(src, dst)
                        logging.info(f"{config_file} → {dst}")
                        break

            # Step 4: Generate deployment manifest
            manifest_path = self._generate_manifest(version, export_dir)
            logging.info(f"Manifest → {manifest_path}")

            # Step 5: Optionally deploy to Pi path (if accessible)
            deploy_path = self.config.deploy_path
            if os.path.exists(os.path.dirname(deploy_path)):
                try:
                    ensure_directory(deploy_path)
                    for f in os.listdir(export_dir):
                        shutil.copy2(
                            os.path.join(export_dir, f),
                            os.path.join(deploy_path, f),
                        )
                    logging.info(f"Deployed to Pi path: {deploy_path}")
                except PermissionError:
                    logging.warning(
                        f"Cannot deploy to {deploy_path} — permission denied. "
                        f"Copy manually: cp {export_dir}/* {deploy_path}/"
                    )

            artifact = ModelPusherArtifact(
                deployed_model_path=tflite_dest,
                manifest_path=manifest_path,
                model_version=version,
            )

            logging.info(f"Model Pusher artifact: {artifact}")
            logging.info("Stage 6: MODEL PUSHER completed")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
