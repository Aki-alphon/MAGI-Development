import re

with open("magi_vision/components/model_trainer.py", "r") as f:
    content = f.read()

# 1. Insert _train_epoch_by_epoch before initiate_model_training
new_method = """    def _train_epoch_by_epoch(
        self,
        total_epochs: int,
        patience: int,
        class_weights: dict,
        best_model_path: str,
        phase_name: str,
        base_lr: float,
        train_ds_fn,
        val_ds_fn,
        steps_per_epoch: int,
        is_cosine_phase: bool = False,
        total_cosine_steps: int = 0,
        initial_model: keras.Model = None,
    ) -> tuple:
        combined_history = {}
        best_val_acc = 0.0
        patience_counter = 0

        # Save the initial model state
        latest_ckpt = best_model_path.replace(".keras", "_latest.keras")
        initial_model.save(latest_ckpt)
        
        # Explicitly delete the initial model from RAM
        del initial_model
        gc.collect()
        tf.keras.backend.clear_session()
        
        for epoch in range(total_epochs):
            logging.info(f"[{phase_name}] ── Epoch {epoch+1}/{total_epochs} (Best: {best_val_acc:.4f}, Patience: {patience_counter}/{patience})")
            
            # 1. Load latest model
            model = keras.models.load_model(latest_ckpt)
            
            # 2. Recompile with correct LR
            if is_cosine_phase:
                elapsed_steps = epoch * steps_per_epoch
                cosine_progress = min(elapsed_steps / max(1, total_cosine_steps), 1.0)
                import math
                current_lr = base_lr * 0.5 * (1 + math.cos(math.pi * cosine_progress))
            else:
                current_lr = base_lr
                
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=current_lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
                steps_per_execution=STEPS_PER_EXECUTION,
            )
            
            # 3. Build dataset for this single epoch
            train_ds = train_ds_fn()
            val_ds = val_ds_fn()
            
            # 4. Train 1 epoch
            history_obj = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epoch+1,
                initial_epoch=epoch,
                class_weight=class_weights,
                verbose=1
            )
            history = history_obj.history
            
            for key, vals in history.items():
                combined_history.setdefault(key, []).extend([float(v) for v in vals])
                
            val_acc = max(history.get("val_accuracy", [0]))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model.save(best_model_path)
                logging.info(f"[{phase_name}] ✓ New best val_acc={best_val_acc:.4f} saved.")
            else:
                patience_counter += 1
                
            # 5. Save latest state
            model.save(latest_ckpt)
            
            # 6. DESTROY EVERYTHING
            del train_ds
            del val_ds
            del history_obj
            del history
            del model
            gc.collect()
            tf.keras.backend.clear_session()
            gc.collect()
            
            if patience_counter >= patience:
                logging.info(f"[{phase_name}] Early stopping at epoch {epoch+1}.")
                break
                
        # Return best model reloaded
        best_model = keras.models.load_model(best_model_path)
        return combined_history, best_val_acc, best_model

    def initiate_model_training(self) -> ModelTrainerArtifact:"""
content = content.replace("    def initiate_model_training(self) -> ModelTrainerArtifact:", new_method)

# 2. Modify Phase 1 and 2 logic in initiate_model_training
phase_replacement = """            # Dataset factory functions so datasets can be built per-epoch
            def get_train_ds():
                return self._create_tf_dataset(
                    self.transformation_artifact.train_npy_dir,
                    self.config.batch_size,
                    augment=True,
                    shuffle=True,
                )
            
            def get_val_ds():
                return self._create_tf_dataset(
                    self.transformation_artifact.val_npy_dir,
                    self.config.batch_size,
                )

            # Compute class weights (use original train_dir to match distribution)
            class_weights = self._compute_class_weights(
                self.ingestion_artifact.train_dir
            )

            # ============ PHASE 1: Train Head Only ============
            logging.info("=" * 40)
            logging.info("PHASE 1: Training classification head (backbone frozen)")
            
            try:
                tf.config.optimizer.set_jit(True)
            except Exception:
                pass

            phase1_ckpt = self.config.trained_model_path.replace(".keras", "_phase1_best.keras")
            steps_per_epoch = self.transformation_artifact.num_train_samples // self.config.batch_size
            
            history1, phase1_acc, model = self._train_epoch_by_epoch(
                total_epochs=self.config.phase1_epochs,
                patience=self.config.phase1_epochs // 2,
                class_weights=class_weights,
                best_model_path=phase1_ckpt,
                phase_name="Phase1",
                base_lr=self.config.phase1_lr,
                train_ds_fn=get_train_ds,
                val_ds_fn=get_val_ds,
                steps_per_epoch=steps_per_epoch,
                is_cosine_phase=False,
                initial_model=model,
            )
            logging.info(f"Phase 1 best val accuracy: {phase1_acc:.4f}")

            # ============ PHASE 2: Fine-tune Top Layers ============
            logging.info("=" * 40)
            logging.info("PHASE 2: Fine-tuning backbone top layers")
            
            base_model = model.get_layer("mobilenetv2_backbone")
            for layer in base_model.layers[self.config.unfreeze_from_layer:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True

            total_phase2_steps = self.config.phase2_epochs * steps_per_epoch

            history2, phase2_acc, model = self._train_epoch_by_epoch(
                total_epochs=self.config.phase2_epochs,
                patience=7,
                class_weights=class_weights,
                best_model_path=self.config.trained_model_path,
                phase_name="Phase2",
                base_lr=self.config.phase2_lr,
                train_ds_fn=get_train_ds,
                val_ds_fn=get_val_ds,
                steps_per_epoch=steps_per_epoch,
                is_cosine_phase=True,
                total_cosine_steps=total_phase2_steps,
                initial_model=model,
            )
            logging.info(f"Phase 2 best val accuracy: {phase2_acc:.4f}")

            # ============ Evaluate on Test Set ============
            test_ds = self._create_tf_dataset(
                self.transformation_artifact.test_npy_dir,
                self.config.batch_size,
                cache=True,
            )"""

pattern = re.compile(r"            # Build datasets — val/test are cached in RAM[\s\S]*?# ============ Evaluate on Test Set ============[\s\S]*?test_ds = self\._create_tf_dataset\([\s\S]*?cache=True,   # small test set — cache it for fast predict loop\n            \)")

content = pattern.sub(phase_replacement, content)

with open("magi_vision/components/model_trainer.py", "w") as f:
    f.write(content)
print("Epoch-by-epoch training loop patched.")
