import os
import sys
import json
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, callbacks, mixed_precision

from magi_vision.entity.config_entity import ModelTrainerConfig
from magi_vision.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from magi_vision.entity.estimator import SpectralPreprocessor
from magi_vision.exception import MAGIVisionException
from magi_vision.logger import logging
from magi_vision.utils.main_utils import (
    read_json_file,
    write_json_file,
    ensure_directory,
)


class ModelTrainer:
    """
    Stage 4: Model Trainer
    ----------------------
    Builds and trains a modified MobileNetV2 with 6-channel input
    for canopy-level plant health classification.

    Two-phase training:
      Phase 1: Frozen backbone, train classification head (15 epochs)
      Phase 2: Unfreeze top layers, fine-tune (30 epochs)

    Exports both Keras (.keras) and TFLite (.tflite) models.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.ingestion_artifact = data_ingestion_artifact
        self.transformation_artifact = data_transformation_artifact
        self.config = model_trainer_config

    def _build_model(self) -> keras.Model:
        """
        Build modified MobileNetV2 with 6-channel input.

        Architecture:
          Input (224, 224, 6) → Channel Adapter Conv2D (6→3) → 
          MobileNetV2 backbone → GAP → Dense(256) → Dropout(0.4) → 
          Dense(NUM_CLASSES, softmax)
        """
        logging.info("Building 6-channel MobileNetV2 model...")

        input_shape = (*self.config.image_size, self.config.input_channels)
        inputs = keras.Input(shape=input_shape, name="spectral_input")

        # Channel adapter: 6ch → 3ch for MobileNetV2 compatibility
        # Uses a 1×1 convolution to learn the optimal channel combination
        x = layers.Conv2D(
            3, (1, 1), padding="same", name="channel_adapter",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.BatchNormalization(name="channel_adapter_bn")(x)
        x = layers.ReLU(name="channel_adapter_relu")(x)

        # MobileNetV2 backbone (ImageNet pretrained)
        base_model = applications.MobileNetV2(
            input_shape=(self.config.image_size[0], self.config.image_size[1], 3),
            include_top=False,
            weights="imagenet",
        )
        base_model._name = "mobilenetv2_backbone"

        # Freeze backbone initially (Phase 1)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model(x, training=False)

        # Classification head
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(
            self.config.dense_units, activation="relu", name="fc1"
        )(x)
        x = layers.Dropout(self.config.dropout_rate, name="dropout")(x)
        outputs = layers.Dense(
            self.config.num_classes,
            activation="softmax",
            dtype="float32",  # Always float32 for output stability
            name="predictions",
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="melchior_v2")

        logging.info(
            f"Model built: {model.count_params():,} total params, "
            f"input shape: {input_shape}"
        )

        return model, base_model

    def _create_tf_dataset(
        self,
        image_dir: str,
        norm_stats: dict,
        batch_size: int,
        augment: bool = False,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset that loads images and applies
        spectral masking preprocessing on-the-fly.
        """
        preprocessor = SpectralPreprocessor(
            image_size=self.config.image_size,
            computed_stats=norm_stats,
        )

        class_names = sorted(os.listdir(image_dir))
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        image_paths = []
        labels = []

        for class_name in class_names:
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    image_paths.append(fpath)
                    labels.append(class_to_idx[class_name])

        def load_and_preprocess(path_bytes, label):
            """TF function to load and preprocess a single image."""
            path_str = path_bytes.numpy().decode("utf-8")

            img = cv2.imread(path_str)
            if img is None:
                # Return zeros if image can't be loaded
                return np.zeros(
                    (*self.config.image_size, self.config.input_channels),
                    dtype=np.float32,
                ), label

            tensor = preprocessor.preprocess(img)

            if augment:
                # Random horizontal flip
                if np.random.random() > 0.5:
                    tensor = np.flip(tensor, axis=1)
                # Random brightness
                brightness = np.random.uniform(0.8, 1.2)
                tensor[:, :, :3] *= brightness
                # Random noise on computed channels
                noise = np.random.normal(0, 0.02, tensor[:, :, 3:].shape)
                tensor[:, :, 3:] += noise.astype(np.float32)

            return tensor.astype(np.float32), label

        def tf_load_preprocess(path, label):
            tensor, lbl = tf.py_function(
                load_and_preprocess,
                [path, label],
                [tf.float32, tf.int32],
            )
            tensor.set_shape(
                [self.config.image_size[0], self.config.image_size[1],
                 self.config.input_channels]
            )
            lbl = tf.one_hot(lbl, self.config.num_classes)
            return tensor, lbl

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, labels)
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        dataset = dataset.map(
            tf_load_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _compute_class_weights(self, train_dir: str) -> dict:
        """Compute class weights for imbalanced datasets."""
        from magi_vision.utils.main_utils import get_class_distribution

        dist = get_class_distribution(train_dir)
        total = sum(dist.values())
        n_classes = len(dist)
        weights = {}

        for i, (class_name, count) in enumerate(sorted(dist.items())):
            if count > 0:
                weights[i] = total / (n_classes * count)
            else:
                weights[i] = 1.0

        logging.info(f"Class weights: {weights}")
        return weights

    def _convert_to_tflite(
        self, model: keras.Model, output_path: str
    ) -> float:
        """Convert Keras model to TFLite with float16 quantization."""
        logging.info("Converting model to TFLite (float16)...")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        ensure_directory(os.path.dirname(output_path))
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"TFLite model saved: {output_path} ({size_mb:.2f} MB)")
        return size_mb

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """Execute model training pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 4: MODEL TRAINING started")
        logging.info("=" * 60)

        try:
            ensure_directory(self.config.trained_model_dir)

            # Enable mixed precision for GPU training
            try:
                mixed_precision.set_global_policy("mixed_float16")
                logging.info("Mixed precision (float16) enabled")
            except Exception:
                logging.info("Mixed precision not available, using float32")

            # Load normalization stats
            norm_stats = read_json_file(
                self.transformation_artifact.normalization_stats_path
            )

            # Build model
            model, base_model = self._build_model()

            # Create datasets
            logging.info("Creating training datasets...")
            train_ds = self._create_tf_dataset(
                self.ingestion_artifact.train_dir,
                norm_stats,
                self.config.batch_size,
                augment=True,
                shuffle=True,
            )
            val_ds = self._create_tf_dataset(
                self.ingestion_artifact.val_dir,
                norm_stats,
                self.config.batch_size,
            )

            # Compute class weights
            class_weights = self._compute_class_weights(
                self.ingestion_artifact.train_dir
            )

            # ============ PHASE 1: Train Head Only ============
            logging.info("=" * 40)
            logging.info("PHASE 1: Training classification head")
            logging.info(
                f"  Epochs: {self.config.phase1_epochs}, "
                f"LR: {self.config.phase1_lr}"
            )
            logging.info("=" * 40)

            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.config.phase1_lr
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            phase1_callbacks = [
                callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=5,
                    restore_best_weights=True,
                ),
            ]

            history1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.phase1_epochs,
                class_weight=class_weights,
                callbacks=phase1_callbacks,
            )

            phase1_acc = max(history1.history.get("val_accuracy", [0]))
            logging.info(f"Phase 1 best val accuracy: {phase1_acc:.4f}")

            # ============ PHASE 2: Fine-tune Top Layers ============
            logging.info("=" * 40)
            logging.info("PHASE 2: Fine-tuning backbone")
            logging.info(
                f"  Unfreezing from layer {self.config.unfreeze_from_layer}, "
                f"Epochs: {self.config.phase2_epochs}, "
                f"LR: {self.config.phase2_lr}"
            )
            logging.info("=" * 40)

            # Unfreeze top layers of backbone
            for layer in base_model.layers[self.config.unfreeze_from_layer:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.config.phase2_lr
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            phase2_callbacks = [
                callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=7,
                    restore_best_weights=True,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                ),
                callbacks.ModelCheckpoint(
                    self.config.trained_model_path,
                    monitor="val_accuracy",
                    save_best_only=True,
                ),
            ]

            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.phase2_epochs,
                class_weight=class_weights,
                callbacks=phase2_callbacks,
            )

            # Load best model from Phase 2
            model = keras.models.load_model(self.config.trained_model_path)
            phase2_acc = max(history2.history.get("val_accuracy", [0]))
            logging.info(f"Phase 2 best val accuracy: {phase2_acc:.4f}")

            # ============ Evaluate on Test Set ============
            test_ds = self._create_tf_dataset(
                self.ingestion_artifact.test_dir,
                norm_stats,
                self.config.batch_size,
            )

            test_results = model.evaluate(test_ds, verbose=0)
            test_accuracy = test_results[1]
            logging.info(f"Test accuracy: {test_accuracy:.4f}")

            # ============ Compute Detailed Metrics ============
            y_true = []
            y_pred = []
            for batch_x, batch_y in test_ds:
                preds = model.predict(batch_x, verbose=0)
                y_pred.extend(np.argmax(preds, axis=1))
                y_true.extend(np.argmax(batch_y.numpy(), axis=1))

            from sklearn.metrics import (
                f1_score, precision_score, recall_score
            )

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            macro_precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            macro_recall = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )

            per_class_f1_arr = f1_score(
                y_true, y_pred, average=None, zero_division=0
            )
            per_class_f1 = {
                self.config.class_names[i]: float(per_class_f1_arr[i])
                for i in range(min(len(per_class_f1_arr), len(self.config.class_names)))
            }

            logging.info(
                f"Test metrics: F1={macro_f1:.4f}, "
                f"Precision={macro_precision:.4f}, "
                f"Recall={macro_recall:.4f}"
            )
            logging.info(f"Per-class F1: {per_class_f1}")

            # ============ TFLite Conversion ============
            self._convert_to_tflite(model, self.config.tflite_model_path)

            # ============ Save Training History ============
            full_history = {}
            for key in history1.history:
                full_history[f"phase1_{key}"] = [
                    float(v) for v in history1.history[key]
                ]
            for key in history2.history:
                full_history[f"phase2_{key}"] = [
                    float(v) for v in history2.history[key]
                ]
            full_history["test_accuracy"] = float(test_accuracy)
            full_history["test_f1"] = float(macro_f1)

            write_json_file(
                self.config.training_history_path, full_history, replace=True
            )

            # ============ Save Class Mapping ============
            class_mapping = {
                str(i): name for i, name in enumerate(self.config.class_names)
            }
            write_json_file(
                self.config.class_mapping_path, class_mapping, replace=True
            )

            metric_artifact = ClassificationMetricArtifact(
                accuracy=float(test_accuracy),
                f1_score=float(macro_f1),
                precision=float(macro_precision),
                recall=float(macro_recall),
                per_class_f1=per_class_f1,
            )

            artifact = ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                tflite_model_path=self.config.tflite_model_path,
                training_history_path=self.config.training_history_path,
                class_mapping_path=self.config.class_mapping_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model Trainer artifact: {artifact}")
            logging.info("Stage 4: MODEL TRAINING completed")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
