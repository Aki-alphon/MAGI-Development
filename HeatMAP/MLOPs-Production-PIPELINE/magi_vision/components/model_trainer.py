import os
import sys
import gc
import math
import json
import glob
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
from magi_vision.constants import STEPS_PER_EXECUTION


class ModelTrainer:
    """
    Stage 4: Model Trainer
    ----------------------
    Builds and trains a modified MobileNetV2 with 8-channel input
    for canopy-level plant health classification.

    Two-phase training:
      Phase 1: Frozen backbone, train classification head only
               Callbacks: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
      Phase 2: Unfreeze top layers, cosine-decay fine-tune
               Callbacks: EarlyStopping + ModelCheckpoint + LearningRateScheduler(cosine)

    Each phase uses a single model.fit() call — the model stays in GPU memory
    the entire time. No per-epoch save/load/clear_session overhead.

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

    # ─────────────────────────────────────────────────────────────────────────
    # Model Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_model(self) -> tuple:
        """
        Build modified MobileNetV2 with 8-channel input.

        Architecture:
          Input (224, 224, 8) → Channel Adapter Conv2D (8→3) →
          MobileNetV2 backbone → GAP → Dense(256) → Dropout(0.4) →
          Dense(NUM_CLASSES, softmax)

        Returns:
            (model, base_model) tuple — base_model reference needed for Phase 2 unfreezing.
        """
        logging.info("Building 8-channel MobileNetV2 model...")

        input_shape = (*self.config.image_size, self.config.input_channels)
        inputs = keras.Input(shape=input_shape, name="spectral_input")

        # Channel adapter: 8ch → 3ch for MobileNetV2 compatibility.
        # 1×1 conv learns the optimal channel combination end-to-end.
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

        # Freeze entire backbone for Phase 1
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
            dtype="float32",   # Always float32 for output stability with mixed precision
            name="predictions",
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="celebi_v2")

        logging.info(
            f"Model built: {model.count_params():,} total params, "
            f"input shape: {input_shape}"
        )

        return model, base_model

    # ─────────────────────────────────────────────────────────────────────────
    # Data Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _create_tf_dataset(
        self,
        npy_dir: str,
        batch_size: int,
        augment: bool = False,
        shuffle: bool = False,
        cache_path: str = None,
    ) -> tf.data.Dataset:
        """
        Build a RAM-safe tf.data pipeline from sharded TFRecords.

        Memory budget (free T4: 12.7 GB system RAM, 15 GB VRAM):
          Each 8-ch float32 image = 224×224×8×4 = 1.53 MB.
          With conservative settings the pipeline holds ~400 MB in system RAM.

        Key choices:
          - num_parallel_calls=2   fixed workers (AUTOTUNE floods RAM on large images)
          - prefetch(2)            fixed 2 batches (AUTOTUNE buffers 8-16 batches = 1.5 GB+)
          - shuffle_buffer=256     256×1.53 MB = ~393 MB (AUTOTUNE had 1024 = 1.57 GB)
          - cache_path (disk)      val/test cached on Colab's 112 GB disk, NOT in RAM
          - drop_remainder=True    fixed batch shape → XLA reuses compiled graph
        """
        # Support both sharded (new) and single-file (legacy) TFRecord layouts
        shard_pattern = os.path.join(npy_dir, "dataset_*-of-*.tfrecord")
        shard_files   = sorted(glob.glob(shard_pattern))

        if shard_files:
            logging.info(f"Loading {len(shard_files)} TFRecord shards from {npy_dir}")
            dataset = tf.data.Dataset.from_tensor_slices(shard_files)
            dataset = dataset.interleave(
                tf.data.TFRecordDataset,
                cycle_length=min(4, len(shard_files)),
                num_parallel_calls=2,       # Fixed — AUTOTUNE over-spawns threads on free T4
                deterministic=False,
            )
        else:
            # Fall back to the legacy single-file layout
            legacy_path = os.path.join(npy_dir, "dataset.tfrecord")
            logging.info(f"Loading legacy single TFRecord: {legacy_path}")
            dataset = tf.data.TFRecordDataset(legacy_path)  # Single reader is fine

        def _parse(example_proto):
            feature_desc = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
            }
            parsed = tf.io.parse_single_example(example_proto, feature_desc)
            # Decode float16 bytes → cast to float32 for training stability
            image = tf.io.decode_raw(parsed["image"], tf.float16)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, [*self.config.image_size, self.config.input_channels])
            label = tf.cast(parsed["label"], tf.int32)
            label = tf.one_hot(label, self.config.num_classes)
            return image, label

        dataset = dataset.map(_parse, num_parallel_calls=2)  # Fixed — AUTOTUNE floods RAM

        # Disk cache (not RAM cache) for val/test.
        # Caching parsed float32 tensors to Colab's 112 GB disk eliminates repeated
        # TFRecord decode + float cast on every epoch — zero system RAM cost.
        if cache_path is not None:
            ensure_directory(cache_path)
            dataset = dataset.cache(os.path.join(cache_path, "tfcache"))
            logging.info(f"Dataset disk-cached at: {cache_path}")

        if shuffle:
            # 256 buffer = 393 MB — large enough for mixing, small enough not to OOM.
            # (Previous value of 1024 added 1.57 GB — primary cause of session crash.)
            dataset = dataset.shuffle(buffer_size=256, reshuffle_each_iteration=True)

        def _augment(tensor, lbl):
            """
            On-device augmentation for training batches.

            Strategy:
              - Geometric transforms (flip) applied to ALL channels together
              - Colour jitter (brightness/contrast/hue) applied to RGB only
                (spectral indices are band ratios — colour shift would corrupt them)
              - Additive Gaussian noise on spectral channels only
            """
            # ── Geometric: applied to full 8-channel tensor ──
            tensor = tf.image.random_flip_left_right(tensor)
            tensor = tf.image.random_flip_up_down(tensor)

            # Split channels after geometric transforms
            rgb      = tensor[:, :, :3]
            spectral = tensor[:, :, 3:]

            # ── Colour jitter: RGB only ──
            rgb = tf.image.random_brightness(rgb, max_delta=0.15)
            rgb = tf.image.random_contrast(rgb, lower=0.8, upper=1.2)
            rgb = tf.image.random_hue(rgb, max_delta=0.04)
            rgb = tf.clip_by_value(rgb, -3.0, 3.0)  # Stay inside ImageNet norm range

            # ── Additive noise: spectral channels only ──
            noise = tf.random.normal(
                shape=tf.shape(spectral), mean=0.0, stddev=0.03
            )
            spectral = spectral + noise

            return tf.concat([rgb, spectral], axis=-1), lbl

        if augment:
            dataset = dataset.map(_augment, num_parallel_calls=2)  # Fixed — AUTOTUNE floods RAM

        # drop_remainder=True → fixed-shape batches → XLA reuses the compiled graph
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # prefetch(2): pre-loads exactly 2 batches while GPU processes current batch.
        # AUTOTUNE here was pre-loading 8-16 batches (up to 1.5 GB) and crashing the session.
        dataset = dataset.prefetch(2)

        return dataset

    # ─────────────────────────────────────────────────────────────────────────
    # Class Weights
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_class_weights(self, train_dir: str) -> dict:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        from magi_vision.utils.main_utils import get_class_distribution

        dist = get_class_distribution(train_dir)
        total = sum(dist.values())
        n_classes = len(dist)
        weights = {}

        for i, (class_name, count) in enumerate(sorted(dist.items())):
            weights[i] = total / (n_classes * count) if count > 0 else 1.0

        logging.info(f"Class weights: {weights}")
        return weights

    # ─────────────────────────────────────────────────────────────────────────
    # TFLite Export
    # ─────────────────────────────────────────────────────────────────────────

    def _convert_to_tflite(
        self, model: keras.Model, output_path: str
    ) -> float:
        """
        Convert Keras model to TFLite with float16 quantization.

        Why clone_model() fails with mixed precision:
          clone_model() copies the layer *configs* from the original model,
          which includes each layer's stored dtype policy (mixed_float16).
          Even after setting the global policy to float32 and casting weights,
          the cloned graph still has float16 compute dtype baked in — so the
          TFLite MLIR converter sees f16 tensors everywhere and raises
          ERROR_NEEDS_FLEX_OPS for every op (Conv2D, Relu, BiasAdd, etc.).

        Fix — build a brand new model under float32 policy:
          _build_model() constructs fresh Keras layers that inherit the CURRENT
          global policy (float32, set just before the call). These layers have
          float32 compute dtype from scratch — no mixed_float16 config leaked.
          Weights are then copied in explicitly as float32 numpy arrays.
          The TFLite converter then sees a fully float32 graph and applies its
          own float16 quantization at conversion time, which is the correct way.
        """
        logging.info("Converting model to TFLite (float16 quantization)...")

        # Temporarily switch to float32 policy so the fresh model is built
        # with float32 compute dtype on all layers
        old_policy = mixed_precision.global_policy()
        mixed_precision.set_global_policy("float32")

        # Build a completely FRESH float32 model — do NOT use clone_model().
        # clone_model copies layer dtype configs (compute_dtype='float16') from
        # the mixed precision original. _build_model() under float32 policy
        # creates genuinely float32 layers.
        logging.info("Building fresh float32 model for TFLite export...")
        model_fp32, _ = self._build_model()

        # Copy trained weights cast explicitly to float32.
        # model.get_weights() returns fp16 numpy arrays from mixed_float16 training.
        fp32_weights = [np.array(w, dtype=np.float32) for w in model.get_weights()]
        model_fp32.set_weights(fp32_weights)
        del fp32_weights

        # Convert: the graph is now entirely float32 — TFLite applies its own
        # float16 quantization at this step (no pre-existing fp16 ops to block).
        converter = tf.lite.TFLiteConverter.from_keras_model(model_fp32)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        # Restore original mixed precision policy
        mixed_precision.set_global_policy(old_policy)

        ensure_directory(os.path.dirname(output_path))
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        del model_fp32
        gc.collect()

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"TFLite model saved: {output_path} ({size_mb:.2f} MB)")
        return size_mb

    # ─────────────────────────────────────────────────────────────────────────
    # Training Phase Runner
    # ─────────────────────────────────────────────────────────────────────────

    def _run_phase(
        self,
        model: keras.Model,
        phase_name: str,
        epochs: int,
        patience: int,
        lr: float,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        best_ckpt_path: str,
        steps_per_epoch: int,
        cosine_decay: bool = False,
        total_cosine_steps: int = 0,
        class_weights: dict = None,
    ) -> dict:
        """
        Run one training phase with a single model.fit() call.

        The model stays in GPU memory for the entire phase. Checkpointing,
        early stopping, and LR scheduling are handled by Keras callbacks
        entirely within the C++ training loop — no Python overhead per epoch.

        Phase 1 (cosine_decay=False):
          Fixed LR + ReduceLROnPlateau as a safety net.

        Phase 2 (cosine_decay=True):
          LearningRateScheduler with cosine annealing from base_lr → ~0.

        Args:
            model:              The Keras model to train (modified in-place).
            phase_name:         Label for logging.
            epochs:             Max epochs for this phase.
            patience:           EarlyStopping patience.
            lr:                 Initial learning rate.
            train_ds:           Training tf.data.Dataset.
            val_ds:             Validation tf.data.Dataset (cached).
            best_ckpt_path:     Path to save best checkpoint.
            steps_per_epoch:    Used only for cosine schedule calculation.
            cosine_decay:       If True, apply cosine LR annealing.
            total_cosine_steps: Total training steps for full cosine cycle.
            class_weights:      Optional class weight dict.

        Returns:
            History dict from model.fit().history.
        """
        logging.info(f"[{phase_name}] Compiling — lr={lr:.2e}, "
                     f"steps_per_execution={STEPS_PER_EXECUTION}")

        # ── Build callback list ──
        cb_list = [
            callbacks.ModelCheckpoint(
                filepath=best_ckpt_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,  # Restore best weights on stop
                verbose=1,
            ),
        ]

        if cosine_decay and total_cosine_steps > 0:
            # Cosine annealing: lr decays from base_lr → ~0 over total_cosine_steps.
            # Uses `epoch` index rather than step for simplicity (close enough for
            # fine-tuning where early stopping fires well before full decay).
            def _cosine_lr(epoch: int) -> float:
                elapsed_steps = epoch * steps_per_epoch
                progress = min(elapsed_steps / max(1, total_cosine_steps), 1.0)
                return float(lr * 0.5 * (1.0 + math.cos(math.pi * progress)))

            cb_list.append(
                callbacks.LearningRateScheduler(_cosine_lr, verbose=0)
            )
            logging.info(
                f"[{phase_name}] Cosine LR schedule: "
                f"{lr:.2e} → ~0 over {total_cosine_steps} steps"
            )
        else:
            # Fixed LR with ReduceLROnPlateau as a fallback safety net
            cb_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=max(2, patience // 3),
                    min_lr=1e-7,
                    verbose=1,
                )
            )

        # Compile once per phase — not per epoch
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            steps_per_execution=STEPS_PER_EXECUTION,
        )

        logging.info(
            f"[{phase_name}] Starting fit — max {epochs} epochs, patience={patience}"
        )
        history_obj = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=cb_list,
            verbose=1,
        )

        best_val_acc = max(history_obj.history.get("val_accuracy", [0.0]))
        actual_epochs = len(history_obj.history.get("val_accuracy", []))
        logging.info(
            f"[{phase_name}] Completed {actual_epochs} epochs. "
            f"Best val_accuracy: {best_val_acc:.4f}"
        )

        return history_obj.history

    # ─────────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """Execute the full two-phase model training pipeline."""
        logging.info("=" * 60)
        logging.info("Stage 4: MODEL TRAINING started")
        logging.info("=" * 60)

        try:
            ensure_directory(self.config.trained_model_dir)

            # ── GPU: memory growth prevents full VRAM pre-allocation on free T4 ──
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # Already initialised — safe to skip

            logging.info(f"GPUs available: {[g.name for g in gpus] or 'None (CPU mode)'}")

            # Suppress verbose cuDNN / XLA INFO spam from C++ layer
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

            # Enable mixed precision for GPU training
            try:
                mixed_precision.set_global_policy("mixed_float16")
                logging.info("Mixed precision (float16) enabled")
            except Exception:
                logging.info("Mixed precision not available, using float32")

            # Load normalization stats (kept for artifact traceability)
            norm_stats = read_json_file(
                self.transformation_artifact.normalization_stats_path
            )

            # ── Build model ──
            model, base_model = self._build_model()

            # ── Build datasets ──
            # Train: augmented + shuffled. Never cached — shuffle requires live iteration.
            # Val:   disk-cached after first pass → skips TFRecord decode on every epoch.
            #        Disk cache NOT RAM cache — 112 GB Colab disk vs 12.7 GB system RAM.
            logging.info("Building tf.data pipelines...")
            val_cache_dir = os.path.join(
                self.config.trained_model_dir, ".val_tfcache"
            )
            train_ds = self._create_tf_dataset(
                self.transformation_artifact.train_npy_dir,
                self.config.batch_size,
                augment=True,
                shuffle=True,
            )
            val_ds = self._create_tf_dataset(
                self.transformation_artifact.val_npy_dir,
                self.config.batch_size,
                cache_path=val_cache_dir,   # ← Disk cache, not RAM cache
            )

            # Compute class weights
            class_weights = self._compute_class_weights(
                self.ingestion_artifact.train_dir
            )

            steps_per_epoch = (
                self.transformation_artifact.num_train_samples // self.config.batch_size
            )
            logging.info(f"Steps per epoch: {steps_per_epoch}")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 1: Train Classification Head (Backbone Frozen)
            # ══════════════════════════════════════════════════════════════════
            logging.info("=" * 40)
            logging.info("PHASE 1: Training classification head (backbone frozen)")

            # Disable XLA JIT for Phase 1 — frozen backbone has non-XLA-friendly ops
            try:
                tf.config.optimizer.set_jit(False)
            except Exception:
                pass

            phase1_ckpt = self.config.trained_model_path.replace(
                ".keras", "_phase1_best.keras"
            )

            history1 = self._run_phase(
                model=model,
                phase_name="Phase1",
                epochs=self.config.phase1_epochs,
                patience=self.config.phase1_epochs // 2,
                lr=self.config.phase1_lr,
                train_ds=train_ds,
                val_ds=val_ds,
                best_ckpt_path=phase1_ckpt,
                steps_per_epoch=steps_per_epoch,
                cosine_decay=False,
                class_weights=class_weights,
            )

            phase1_acc = max(history1.get("val_accuracy", [0.0]))
            logging.info(f"Phase 1 complete — best val_accuracy: {phase1_acc:.4f}")

            # ══════════════════════════════════════════════════════════════════
            # PHASE 2: Fine-tune Top Backbone Layers (Cosine LR Decay)
            # ══════════════════════════════════════════════════════════════════
            logging.info("=" * 40)
            logging.info("PHASE 2: Fine-tuning backbone top layers")

            # Unfreeze top backbone layers — keep BatchNorm frozen (critical for fine-tuning
            # stability: prevents domain shift from small fine-tune batches resetting BN stats)
            backbone_name = next(
                l.name for l in model.layers if "mobilenet" in l.name.lower()
            )
            base_model = model.get_layer(backbone_name)
            unfrozen = 0
            for layer in base_model.layers[self.config.unfreeze_from_layer:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
                    unfrozen += 1

            total_trainable = sum(1 for l in model.layers if l.trainable)
            logging.info(
                f"Phase 2: {unfrozen} backbone layers unfrozen, "
                f"{total_trainable} total trainable layers"
            )

            # Enable XLA JIT for Phase 2 — all ops now in the fine-tune graph
            try:
                tf.config.optimizer.set_jit(True)
                logging.info("XLA JIT enabled for Phase 2")
            except Exception:
                pass

            total_phase2_steps = self.config.phase2_epochs * steps_per_epoch

            history2 = self._run_phase(
                model=model,
                phase_name="Phase2",
                epochs=self.config.phase2_epochs,
                patience=7,
                lr=self.config.phase2_lr,
                train_ds=train_ds,
                val_ds=val_ds,
                best_ckpt_path=self.config.trained_model_path,
                steps_per_epoch=steps_per_epoch,
                cosine_decay=True,
                total_cosine_steps=total_phase2_steps,
                class_weights=class_weights,
            )

            phase2_acc = max(history2.get("val_accuracy", [0.0]))
            logging.info(f"Phase 2 complete — best val_accuracy: {phase2_acc:.4f}")

            # ══════════════════════════════════════════════════════════════════
            # EVALUATE on Test Set
            # ══════════════════════════════════════════════════════════════════
            test_cache_dir = os.path.join(
                self.config.trained_model_dir, ".test_tfcache"
            )
            test_ds = self._create_tf_dataset(
                self.transformation_artifact.test_npy_dir,
                self.config.batch_size,
                cache_path=test_cache_dir,  # ← Disk cache, not RAM cache
            )

            logging.info("Evaluating on test set...")
            test_results = model.evaluate(test_ds, verbose=1)
            test_accuracy = test_results[1]
            logging.info(f"Test accuracy: {test_accuracy:.4f}")

            # ── Detailed per-class metrics ──
            y_true, y_pred = [], []
            for batch_x, batch_y in test_ds:
                preds = model.predict(batch_x, verbose=0)
                y_pred.extend(np.argmax(preds, axis=1))
                y_true.extend(np.argmax(batch_y.numpy(), axis=1))

            from sklearn.metrics import f1_score, precision_score, recall_score

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            macro_f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
            macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            macro_recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

            per_class_f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
            per_class_f1 = {
                self.config.class_names[i]: float(per_class_f1_arr[i])
                for i in range(min(len(per_class_f1_arr), len(self.config.class_names)))
            }

            logging.info(
                f"Test metrics — F1: {macro_f1:.4f} | "
                f"Precision: {macro_precision:.4f} | "
                f"Recall: {macro_recall:.4f}"
            )
            logging.info(f"Per-class F1: {per_class_f1}")

            # ══════════════════════════════════════════════════════════════════
            # TFLite Conversion
            # ══════════════════════════════════════════════════════════════════
            self._convert_to_tflite(model, self.config.tflite_model_path)

            # ══════════════════════════════════════════════════════════════════
            # Persist Artefacts
            # ══════════════════════════════════════════════════════════════════
            full_history = {}
            for key, vals in history1.items():
                full_history[f"phase1_{key}"] = [float(v) for v in vals]
            for key, vals in history2.items():
                full_history[f"phase2_{key}"] = [float(v) for v in vals]
            full_history["test_accuracy"] = float(test_accuracy)
            full_history["test_f1"]       = float(macro_f1)

            write_json_file(self.config.training_history_path, full_history, replace=True)

            class_mapping = {
                str(i): name for i, name in enumerate(self.config.class_names)
            }
            write_json_file(self.config.class_mapping_path, class_mapping, replace=True)

            # ── Build and return artifact ──
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

            logging.info(f"ModelTrainerArtifact: {artifact}")
            logging.info("Stage 4: MODEL TRAINING completed successfully")
            return artifact

        except Exception as e:
            raise MAGIVisionException(e, sys) from e
