import sys

from magi_vision.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from magi_vision.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from magi_vision.components.data_ingestion import DataIngestion
from magi_vision.components.data_validation import DataValidation
from magi_vision.components.data_transformation import DataTransformation
from magi_vision.components.model_trainer import ModelTrainer
from magi_vision.components.model_evaluation import ModelEvaluation
from magi_vision.components.model_pusher import ModelPusher
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging


class MAGITrainPipeline:
    """
    MAGI Vision Training Pipeline
    =============================
    Orchestrates the full 6-stage training pipeline:
      1. Data Ingestion    — download/split dataset
      2. Data Validation   — check image integrity & class balance
      3. Data Transformation — compute spectral normalization stats
      4. Model Trainer     — build & train 6ch MobileNetV2 + TFLite export
      5. Model Evaluation  — benchmark TFLite latency & accuracy
      6. Model Pusher      — deploy to export directory
    """

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Stage 1: Data Ingestion"""
        try:
            config = DataIngestionConfig()
            data_ingestion = DataIngestion(data_ingestion_config=config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Stage 2: Data Validation"""
        try:
            config = DataValidationConfig()
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=config,
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """Stage 3: Data Transformation"""
        try:
            config = DataTransformationConfig()
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=config,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def start_model_trainer(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        """Stage 4: Model Trainer"""
        try:
            config = ModelTrainerConfig()
            model_trainer = ModelTrainer(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=config,
            )
            return model_trainer.initiate_model_training()
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def start_model_evaluation(
        self, model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        """Stage 5: Model Evaluation"""
        try:
            config = ModelEvaluationConfig()
            model_evaluation = ModelEvaluation(
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=config,
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def start_model_pusher(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ) -> ModelPusherArtifact:
        """Stage 6: Model Pusher"""
        try:
            config = ModelPusherConfig()
            model_pusher = ModelPusher(
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=config,
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise MAGIVisionException(e, sys) from e

    def run_pipeline(self) -> None:
        """Execute the full training pipeline."""
        logging.info("╔" + "═" * 58 + "╗")
        logging.info("║   MAGI VISION TRAINING PIPELINE — MELCHIOR MODEL        ║")
        logging.info("╚" + "═" * 58 + "╝")

        try:
            # Stage 1
            data_ingestion_artifact = self.start_data_ingestion()

            # Stage 2
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )

            # Stage 3
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact, data_validation_artifact
            )

            # Stage 4
            model_trainer_artifact = self.start_model_trainer(
                data_ingestion_artifact, data_transformation_artifact
            )

            # Stage 5
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact
            )

            # Stage 6
            model_pusher_artifact = self.start_model_pusher(
                model_trainer_artifact, model_evaluation_artifact
            )

            logging.info("╔" + "═" * 58 + "╗")
            logging.info("║   PIPELINE COMPLETE                                     ║")
            logging.info(
                f"║   Model: {model_pusher_artifact.deployed_model_path:<47}║"
            )
            logging.info(
                f"║   Version: {model_pusher_artifact.model_version:<45}║"
            )
            logging.info(
                f"║   Accepted: {str(model_evaluation_artifact.is_model_accepted):<44}║"
            )
            logging.info("╚" + "═" * 58 + "╝")

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
