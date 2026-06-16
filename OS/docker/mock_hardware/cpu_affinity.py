"""
MAGI OS — CPU affinity helper
docker/mock_hardware/cpu_affinity.py

Patches os.sched_setaffinity to be a no-op on non-Linux platforms.
Imported automatically via PYTHONPATH in Docker.
"""

import os
import sys

# Patch only if not on Linux (Docker on Linux still supports it, but
# Docker on Windows/Mac with host isolation does not)
if not hasattr(os, "sched_setaffinity"):
    def _noop_affinity(pid, cores):
        pass
    os.sched_setaffinity = _noop_affinity
