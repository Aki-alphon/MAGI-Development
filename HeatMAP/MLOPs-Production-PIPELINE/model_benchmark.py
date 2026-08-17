"""
MAGI Vision — TFLite Model Benchmark
=====================================
Full-fledged performance test for celebi.tflite / celebi_export.tflite.

Tests:
  1. Model loading & signature validation
  2. Single-inference latency (warm + cold)
  3. Batch throughput (frames/sec)
  4. Pi 4B simulation (4-thread XNNPACK mode)
  5. Confidence distribution across all 4 classes
  6. Memory footprint delta
  7. Consistency check: both TFLite models give identical outputs

Usage:
  python model_benchmark.py
  python model_benchmark.py --model /path/to/custom.tflite
  python model_benchmark.py --runs 200
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import numpy as np

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def g(s): return f"{GREEN}{s}{RESET}"
def y(s): return f"{YELLOW}{s}{RESET}"
def r(s): return f"{RED}{s}{RESET}"
def c(s): return f"{CYAN}{s}{RESET}"
def b(s): return f"{BOLD}{s}{RESET}"

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS_DIR   = Path(__file__).parent.parent.parent / "AIML MODELS"
IMG_H, IMG_W = 224, 224
NUM_CHANNELS  = 8                 # R G B ExG GRVI VARI GLI NGBDI
NUM_CLASSES   = 4
CLASS_NAMES   = ["baseline_healthy", "early_nitrogen_stress",
                 "active_chlorosis", "severe_deficiency"]
HEALTH_SCORES = [1.00, 0.68, 0.35, 0.05]   # per class


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers (mirrors the training pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _load_norm_stats(models_dir: Path) -> Optional[dict]:
    """Load per-channel normalization stats saved during data transformation."""
    for name in ["normalization_stats.json", "normalization_stats (1).json"]:
        p = models_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _build_8ch_tensor(rgb: np.ndarray, norm_stats: Optional[dict]) -> np.ndarray:
    """
    Converts a (H, W, 3) uint8 RGB image to a (H, W, 8) float32 tensor,
    matching the exact pipeline preprocessing:
      ch0 R, ch1 G, ch2 B  — normalised to [0,1]
      ch3 ExG   = 2G - R - B
      ch4 GRVI  = (G-R)/(G+R+ε)
      ch5 VARI  = (G-R)/(G+R-B+ε)
      ch6 GLI   = (2G-R-B)/(2G+R+B+ε)
      ch7 NGBDI = (G-B)/(G+B+ε)
    Then per-channel z-score normalisation using saved stats.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    R, G, B = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    eps = 1e-6

    ExG   = 2*G - R - B
    GRVI  = (G - R) / (G + R + eps)
    VARI  = (G - R) / (G + R - B + eps)
    GLI   = (2*G - R - B) / (2*G + R + B + eps)
    NGBDI = (G - B)        / (G + B + eps)

    tensor = np.stack([R, G, B, ExG, GRVI, VARI, GLI, NGBDI], axis=-1)

    # Apply z-score normalisation for the computed indices
    if norm_stats:
        key_map = {3: "ExG", 4: "GRVI", 5: "VARI", 6: "GLI", 7: "NGBDI"}
        for ch_idx, key in key_map.items():
            if key in norm_stats:
                mean = norm_stats[key]["mean"]
                std  = norm_stats[key]["std"] + eps
                tensor[..., ch_idx] = (tensor[..., ch_idx] - mean) / std

    return tensor


def _make_synthetic_batch(n: int, seed: int = 42,
                           norm_stats: Optional[dict] = None) -> np.ndarray:
    """
    Generates n synthetic 8-channel inputs that cover the full
    physiologically plausible range of plant reflectance values.
    Used when no real test images are available.
    """
    rng = np.random.default_rng(seed)
    batch = []
    for _ in range(n):
        # Simulate different plant health states
        state = rng.integers(0, 4)
        if state == 0:   # healthy — high green, normal red
            R = rng.uniform(0.20, 0.40, (IMG_H, IMG_W)).astype(np.float32)
            G = rng.uniform(0.45, 0.65, (IMG_H, IMG_W)).astype(np.float32)
            B = rng.uniform(0.10, 0.25, (IMG_H, IMG_W)).astype(np.float32)
        elif state == 1: # early N-stress — slightly yellowing
            R = rng.uniform(0.35, 0.55, (IMG_H, IMG_W)).astype(np.float32)
            G = rng.uniform(0.40, 0.55, (IMG_H, IMG_W)).astype(np.float32)
            B = rng.uniform(0.10, 0.22, (IMG_H, IMG_W)).astype(np.float32)
        elif state == 2: # chlorosis — yellow/pale leaves
            R = rng.uniform(0.50, 0.70, (IMG_H, IMG_W)).astype(np.float32)
            G = rng.uniform(0.45, 0.60, (IMG_H, IMG_W)).astype(np.float32)
            B = rng.uniform(0.10, 0.20, (IMG_H, IMG_W)).astype(np.float32)
        else:            # severe — brown/necrotic
            R = rng.uniform(0.55, 0.75, (IMG_H, IMG_W)).astype(np.float32)
            G = rng.uniform(0.30, 0.45, (IMG_H, IMG_W)).astype(np.float32)
            B = rng.uniform(0.05, 0.18, (IMG_H, IMG_W)).astype(np.float32)

        rgb_u8 = (np.stack([R, G, B], axis=-1) * 255).clip(0, 255).astype(np.uint8)
        tensor = _build_8ch_tensor(rgb_u8, norm_stats)
        batch.append(tensor)

    return np.stack(batch, axis=0)   # (n, 224, 224, 8)


# ─────────────────────────────────────────────────────────────────────────────
# TFLite runner
# ─────────────────────────────────────────────────────────────────────────────

def load_interpreter(model_path: str, num_threads: int = 1):
    """Load TFLite model and return (interpreter, input_details, output_details)."""
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(
        model_path=str(model_path),
        num_threads=num_threads,
    )
    interp.allocate_tensors()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()
    return interp, in_det, out_det


def run_single(interp, in_det, out_det, tensor: np.ndarray) -> np.ndarray:
    """Run a single (1, 224, 224, 8) tensor through the interpreter."""
    inp = np.expand_dims(tensor, 0).astype(np.float32)
    interp.set_tensor(in_det[0]["index"], inp)
    interp.invoke()
    return interp.get_tensor(out_det[0]["index"])[0]   # (4,)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark functions
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_latency(interp, in_det, out_det, tensors: np.ndarray,
                      warmup: int = 5, runs: int = 100) -> dict:
    """
    Measure single-image inference latency.
    Returns dict with p50, p90, p95, p99, mean, min, max in ms.
    """
    # Warm-up (JIT / kernel caching)
    for i in range(warmup):
        run_single(interp, in_det, out_det, tensors[i % len(tensors)])

    latencies = []
    for i in range(runs):
        t0 = time.perf_counter()
        run_single(interp, in_det, out_det, tensors[i % len(tensors)])
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms" : float(np.mean(arr)),
        "min_ms"  : float(np.min(arr)),
        "max_ms"  : float(np.max(arr)),
        "p50_ms"  : float(np.percentile(arr, 50)),
        "p90_ms"  : float(np.percentile(arr, 90)),
        "p95_ms"  : float(np.percentile(arr, 95)),
        "p99_ms"  : float(np.percentile(arr, 99)),
        "std_ms"  : float(np.std(arr)),
        "fps"     : 1000.0 / float(np.mean(arr)),
    }


def benchmark_throughput(interp, in_det, out_det,
                         tensors: np.ndarray, batch_size: int = 12) -> dict:
    """
    Simulate real-world canopy scanning: 12-tile grid from one camera frame.
    Returns frames/sec and tiles/sec.
    """
    n = len(tensors)
    tile_times = []
    for start in range(0, n, batch_size):
        chunk = tensors[start:start + batch_size]
        t0 = time.perf_counter()
        for tile in chunk:
            run_single(interp, in_det, out_det, tile)
        tile_times.append((time.perf_counter() - t0) * 1000)

    mean_grid_ms  = float(np.mean(tile_times))
    tiles_per_sec = (batch_size / mean_grid_ms) * 1000
    frames_per_sec = tiles_per_sec / batch_size   # 1 camera frame = 12 tiles
    return {
        "mean_grid_ms"  : mean_grid_ms,
        "tiles_per_sec" : tiles_per_sec,
        "frames_per_sec": frames_per_sec,
    }


def collect_predictions(interp, in_det, out_det,
                        tensors: np.ndarray) -> dict:
    """
    Run full test set, collect class predictions, confidence stats, health scores.
    """
    preds, confidences, health = [], [], []
    for t in tensors:
        probs = run_single(interp, in_det, out_det, t)
        cls   = int(np.argmax(probs))
        conf  = float(np.max(probs))
        hs    = float(np.dot(probs, HEALTH_SCORES))
        preds.append(cls)
        confidences.append(conf)
        health.append(hs)

    preds = np.array(preds)
    counts = {CLASS_NAMES[i]: int(np.sum(preds == i)) for i in range(NUM_CLASSES)}
    return {
        "class_distribution" : counts,
        "mean_confidence"    : float(np.mean(confidences)),
        "min_confidence"     : float(np.min(confidences)),
        "max_confidence"     : float(np.max(confidences)),
        "mean_health_score"  : float(np.mean(health)),
        "predictions"        : preds.tolist(),
        "confidences"        : confidences,
    }


def measure_memory(interp, in_det, out_det, tensor: np.ndarray) -> float:
    """Return peak RAM usage in MB during a single inference."""
    tracemalloc.start()
    run_single(interp, in_det, out_det, tensor)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def check_consistency(models: list, tensors: np.ndarray,
                      tolerance: float = 1e-3) -> dict:
    """Compare outputs of two models on the same inputs."""
    if len(models) < 2:
        return {"skipped": True}

    (i1, in1, out1), (i2, in2, out2) = models[0], models[1]
    diffs = []
    agreements = 0
    n = min(50, len(tensors))
    for t in tensors[:n]:
        p1 = run_single(i1, in1, out1, t)
        p2 = run_single(i2, in2, out2, t)
        diffs.append(float(np.max(np.abs(p1 - p2))))
        if np.argmax(p1) == np.argmax(p2):
            agreements += 1

    return {
        "max_diff"        : float(np.max(diffs)),
        "mean_diff"       : float(np.mean(diffs)),
        "agreement_pct"   : agreements / n * 100,
        "numerically_same": float(np.max(diffs)) < tolerance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

BAR_W = 40

def _bar(val, max_val, width=BAR_W, fill="█", empty="░"):
    filled = int(round(val / max_val * width)) if max_val > 0 else 0
    return fill * filled + empty * (width - filled)


def print_report(model_path: str, model_size_mb: float,
                 lat: dict, tput: dict, pred: dict,
                 mem_mb: float, threads: int) -> None:

    W = 70
    sep = "─" * W

    print(f"\n{b('╔' + '═'*W + '╗')}")
    print(f"{b('║')}  {c('MAGI Vision — Celebi TFLite Benchmark'):<{W-2}}{b('║')}")
    print(f"{b('╚' + '═'*W + '╝')}")

    # ── Model info
    print(f"\n{b('MODEL')}")
    print(sep)
    fname = Path(model_path).name
    print(f"  File          : {c(fname)}")
    print(f"  Size          : {model_size_mb:.2f} MB")
    print(f"  Input shape   : (1, 224, 224, 8)  float32")
    print(f"  Output shape  : (1, 4)  softmax")
    print(f"  Threads used  : {threads}")

    # ── Latency
    print(f"\n{b('LATENCY  (single image inference)')}")
    print(sep)

    # Pi 4B reference: ~125 ms reported in README
    pi_budget = 125.0
    color_fn = g if lat['p95_ms'] <= pi_budget else (y if lat['p95_ms'] <= 200 else r)
    mean_str = f"{lat['mean_ms']:6.1f} ms"
    print(f"  Mean          : {color_fn(mean_str)}  "
          f"| FPS: {lat['fps']:.1f}")
    print(f"  Min / Max     : {lat['min_ms']:.1f} ms  /  {lat['max_ms']:.1f} ms")
    print(f"  p50           : {lat['p50_ms']:.1f} ms")
    print(f"  p90           : {lat['p90_ms']:.1f} ms")
    print(f"  p95           : {lat['p95_ms']:.1f} ms  \u2190 key SLA metric")
    print(f"  p99           : {lat['p99_ms']:.1f} ms")
    print(f"  Std dev       : \u00b1{lat['std_ms']:.1f} ms")


    fit_pi = lat['p95_ms'] <= pi_budget
    verdict = g("✅ FITS Pi 4B real-time budget (<125 ms)") if fit_pi \
              else r(f"⚠️  EXCEEDS Pi 4B budget (>{pi_budget} ms p95)")
    print(f"\n  {verdict}")

    # ── Throughput
    print(f"\n{b('THROUGHPUT  (12-tile canopy grid / camera frame)')}")
    print(sep)
    print(f"  Grid latency  : {tput['mean_grid_ms']:.0f} ms  per camera frame")
    print(f"  Tiles/sec     : {tput['tiles_per_sec']:.1f}")
    print(f"  Frames/sec    : {tput['frames_per_sec']:.2f}  "
          f"{g('(real-time)') if tput['frames_per_sec'] >= 1.0 else y('(sub-realtime)')}")

    # ── Class distribution
    print(f"\n{b('PREDICTION DISTRIBUTION  (synthetic test set)')}")
    print(sep)
    total = sum(pred['class_distribution'].values())
    max_count = max(pred['class_distribution'].values()) or 1
    for cls, cnt in pred['class_distribution'].items():
        pct = cnt / total * 100
        hs  = HEALTH_SCORES[CLASS_NAMES.index(cls)]
        bar = _bar(cnt, max_count)
        hs_color = g if hs >= 0.65 else (y if hs >= 0.30 else r)
        print(f"  {cls:<25} {bar}  {cnt:3d} ({pct:4.1f}%)  "
              f"HS={hs_color(f'{hs:.2f}')}")

    print(f"\n  Mean confidence   : {pred['mean_confidence']*100:.1f}%")
    print(f"  Min confidence    : {pred['min_confidence']*100:.1f}%")
    print(f"  Max confidence    : {pred['max_confidence']*100:.1f}%")
    print(f"  Mean health score : {pred['mean_health_score']:.3f}  "
          f"(0=dead 1=perfect)")

    # ── Memory
    print(f"\n{b('MEMORY')}")
    print(sep)
    print(f"  Peak RAM per inference: {mem_mb:.2f} MB")
    print(f"  Model file on disk    : {model_size_mb:.2f} MB")


def print_consistency(consistency: dict) -> None:
    print(f"\n{b('CONSISTENCY CHECK  (celebi.tflite vs celebi_export.tflite)')}")
    print("─" * 70)
    if consistency.get("skipped"):
        print(f"  {y('Skipped — only one model found')}")
        return
    print(f"  Max output diff   : {consistency['max_diff']:.6f}")
    print(f"  Mean output diff  : {consistency['mean_diff']:.6f}")
    print(f"  Class agreement   : {consistency['agreement_pct']:.1f}%")
    same = consistency['numerically_same']
    print(f"  Numerically same  : {g('YES ✅') if same else y('NO (small numeric drift)')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAGI Vision TFLite Benchmark")
    parser.add_argument("--model",   type=str, default=None,
                        help="Path to a specific .tflite file to test")
    parser.add_argument("--runs",    type=int, default=100,
                        help="Number of latency measurement runs (default: 100)")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads (4 = Pi 4B XNNPACK mode, default: 4)")
    parser.add_argument("--samples", type=int, default=120,
                        help="Number of synthetic test samples (default: 120)")
    args = parser.parse_args()

    # ── Locate models
    model_paths = []
    if args.model:
        model_paths = [Path(args.model)]
    else:
        for name in ["celebi.tflite", "celebi_export.tflite"]:
            p = MODELS_DIR / name
            if p.exists():
                model_paths.append(p)

    if not model_paths:
        print(r(f"❌  No .tflite models found in {MODELS_DIR}"))
        print(f"    Run with: python model_benchmark.py --model /path/to/model.tflite")
        sys.exit(1)

    # ── Load normalization stats
    norm_stats = _load_norm_stats(MODELS_DIR)
    if norm_stats:
        print(g(f"✅ Normalization stats loaded ({len(norm_stats)} channels)"))
    else:
        print(y("⚠️  normalization_stats.json not found — using unnormalized indices"))

    # ── Generate synthetic test data
    print(f"⚙️  Generating {args.samples} synthetic 8-channel test tensors...")
    tensors = _make_synthetic_batch(args.samples, seed=42, norm_stats=norm_stats)
    print(f"   Input shape: {tensors.shape}  dtype: {tensors.dtype}")
    print(f"   Value range: [{tensors.min():.3f}, {tensors.max():.3f}]")

    # ── Run benchmark on each model
    loaded_models = []
    for model_path in model_paths:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*70}")
        print(b(f"Testing: {model_path.name}  ({size_mb:.2f} MB)"))
        print('='*70)

        print(f"  Loading with {args.threads} threads (Pi 4B XNNPACK simulation)...")
        try:
            interp, in_det, out_det = load_interpreter(model_path, args.threads)
        except Exception as e:
            print(r(f"  ❌ Failed to load: {e}"))
            continue

        # Signature check
        in_shape  = in_det[0]["shape"].tolist()
        out_shape = out_det[0]["shape"].tolist()
        in_dtype  = str(in_det[0]["dtype"])
        print(f"  Input  : shape={in_shape}  dtype={in_dtype}")
        print(f"  Output : shape={out_shape}")
        if in_shape[1:] != [IMG_H, IMG_W, NUM_CHANNELS]:
            print(r(f"  ⚠️  Unexpected input shape {in_shape} — expected [1,224,224,8]"))

        loaded_models.append((interp, in_det, out_det))

        # Memory
        print("  Measuring memory...")
        mem_mb = measure_memory(interp, in_det, out_det, tensors[0])

        # Latency
        print(f"  Running {args.runs} latency samples (warmup: 5)...")
        lat = benchmark_latency(interp, in_det, out_det, tensors,
                                warmup=5, runs=args.runs)

        # Throughput
        print("  Running throughput test (12-tile batches)...")
        tput = benchmark_throughput(interp, in_det, out_det, tensors)

        # Predictions
        print(f"  Collecting predictions on all {args.samples} samples...")
        pred = collect_predictions(interp, in_det, out_det, tensors)

        # Print
        print_report(str(model_path), size_mb, lat, tput, pred, mem_mb, args.threads)

    # ── Consistency between the two models
    if len(loaded_models) >= 2:
        print("\nChecking output consistency between both models...")
        consistency = check_consistency(loaded_models, tensors)
        print_consistency(consistency)

    print(f"\n{b('Done.')}  Tested {len(loaded_models)}/{len(model_paths)} model(s).\n")


if __name__ == "__main__":
    main()
