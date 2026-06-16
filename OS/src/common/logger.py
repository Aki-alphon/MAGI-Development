"""
MAGI OS — Centralized Logger
/opt/magi/src/common/logger.py
"""

import logging
import logging.handlers
import os
import sys

LOG_DIR = "/opt/magi/logs"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create a logger with both console and rotating file handlers.

    Args:
        name: Logger name (e.g. 'magi1', 'sensor_hub')
        level: Log level string — DEBUG | INFO | WARNING | ERROR
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # ── Console handler ───────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler (10 MB × 5 files) ───────────
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
