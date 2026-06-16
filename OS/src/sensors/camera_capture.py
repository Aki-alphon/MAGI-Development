"""
MAGI OS — Camera Capture Process
/opt/magi/src/sensors/camera_capture.py

Captures frames from USB/Pi camera and writes them to POSIX
shared memory for zero-copy access by MAGI-1 (Melchior).
Runs on Core 0 alongside sensor_hub.
"""

import os, sys, time, signal, yaml, cv2, numpy as np
sys.path.insert(0, "/opt/magi/src")
try:
    os.sched_setaffinity(0, {0})  # Core 0 (no-op in Docker)
except (AttributeError, OSError):
    pass


from common.logger import get_logger
from common.ipc import SharedFrame

log = get_logger("camera")

with open("/opt/magi/config/config.yaml") as f:
    CFG = yaml.safe_load(f)

CAM_CFG = CFG["camera"]
IPC_CFG = CFG["ipc"]
_running = True

def _shutdown(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

def main():
    if not CAM_CFG.get("enabled", False):
        log.info("Camera disabled in config.yaml — exiting.")
        return

    source = CAM_CFG.get("source", 0)
    width  = CAM_CFG.get("width",  640)
    height = CAM_CFG.get("height", 480)
    fps    = CAM_CFG.get("fps",    30)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimize latency

    if not cap.isOpened():
        log.error(f"Cannot open camera source {source}")
        return

    log.info(f"Camera opened: {width}x{height} @ {fps} fps")

    shm = SharedFrame(
        name=IPC_CFG["shm_camera"],
        size=IPC_CFG["shm_camera_size"],
        create=True
    )

    target_dt = 1.0 / fps
    n = 0

    while _running:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        shm.write(frame)
        n += 1
        if n % (fps * 30) == 0:
            log.info(f"Camera alive — {n} frames captured")

        sleep_t = target_dt - (time.monotonic() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    cap.release()
    shm.close()
    log.info("Camera capture stopped.")

if __name__ == "__main__":
    main()
