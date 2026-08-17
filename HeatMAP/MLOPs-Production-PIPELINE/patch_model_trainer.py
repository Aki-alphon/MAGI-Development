import re

with open("magi_vision/components/model_trainer.py", "r") as f:
    content = f.read()

# 1. Remove _chunked_train_phase entirely
chunk_regex = re.compile(r"    # ── Chunked training ────────────────────────────────────────────────────.*?    def initiate_model_training\(self\) -> ModelTrainerArtifact:", re.DOTALL)
content = chunk_regex.sub("    def initiate_model_training(self) -> ModelTrainerArtifact:", content)

# 2. Replace Phase 1 training
phase1_regex = re.compile(r"            phase1_ckpt = self\.config\.trained_model_path\.replace\([\s\S]*?logging\.info\(f\"Phase 1 best val accuracy: \{phase1_acc:\.4f\}\"\)")
phase1_replacement = """            phase1_ckpt = self.config.trained_model_path.replace(
                ".keras", "_phase1_best.keras"
            )
            
            callbacks_p1 = [
                keras.callbacks.ModelCheckpoint(
                    filepath=phase1_ckpt, save_best_only=True, monitor="val_accuracy"
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=self.config.phase1_epochs // 2, restore_best_weights=True
                )
            ]
            
            history1_obj = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.phase1_epochs,
                class_weight=class_weights,
                callbacks=callbacks_p1,
                verbose=1
            )
            history1 = history1_obj.history
            phase1_acc = max(history1.get("val_accuracy", [0]))
            logging.info(f"Phase 1 best val accuracy: {phase1_acc:.4f}")"""
content = phase1_regex.sub(phase1_replacement, content)

# 3. Replace Phase 2 training
phase2_regex = re.compile(r"            history2, phase2_acc, model = self\._chunked_train_phase\([\s\S]*?logging\.info\(f\"Phase 2 best val accuracy: \{phase2_acc:\.4f\}\"\)")
phase2_replacement = """            callbacks_p2 = [
                keras.callbacks.ModelCheckpoint(
                    filepath=self.config.trained_model_path, save_best_only=True, monitor="val_accuracy"
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=7, restore_best_weights=True
                )
            ]
            
            history2_obj = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.phase2_epochs,
                class_weight=class_weights,
                callbacks=callbacks_p2,
                verbose=1
            )
            history2 = history2_obj.history
            phase2_acc = max(history2.get("val_accuracy", [0]))
            logging.info(f"Phase 2 best val accuracy: {phase2_acc:.4f}")"""
content = phase2_regex.sub(phase2_replacement, content)

# 4. In "Save Training History", make sure we convert floats
history_save_regex = re.compile(r"            for key, vals in history1\.items\(\):\n                full_history\[f\"phase1_\{key\}\"\] = vals\n            for key, vals in history2\.items\(\):\n                full_history\[f\"phase2_\{key\}\"\] = vals")
history_save_replacement = """            for key, vals in history1.items():
                full_history[f"phase1_{key}"] = [float(v) for v in vals]
            for key, vals in history2.items():
                full_history[f"phase2_{key}"] = [float(v) for v in vals]"""
content = history_save_regex.sub(history_save_replacement, content)

with open("magi_vision/components/model_trainer.py", "w") as f:
    f.write(content)
print("Patched successfully")
