# =============================================================================
# MAGI OS — Docker Test Environment
# Base: python:3.11-slim (x86_64 compatible, fast iteration)
# Mocks: pigpio, smbus2, spidev, RPi.GPIO via stub layer
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="MAGI OS"
LABEL description="MAGI OS testing environment with mocked RPi hardware"

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libzmq3-dev \
    libopenblas-dev \
    libopencv-dev \
    python3-dev \
    procps \
    htop \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
COPY docker/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Project structure ─────────────────────────────────────────────────────────
RUN mkdir -p /opt/magi/{models,src,config,logs} \
             /tmp/magi

# ── Copy source ───────────────────────────────────────────────────────────────
COPY src/           /opt/magi/src/
COPY src/common/config.yaml /opt/magi/config/config.yaml

# ── Copy mock hardware layer (overrides real RPi libs in Docker) ──────────────
COPY docker/mock_hardware/ /opt/magi/mock_hardware/

# ── Copy test config (enables all sensors with simulated data) ────────────────
COPY docker/test_config.yaml /opt/magi/config/config.yaml

# ── Copy dummy TFLite models ──────────────────────────────────────────────────
COPY docker/dummy_models/ /opt/magi/models/

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH="/opt/magi/src:/opt/magi/mock_hardware"
ENV MAGI_ENV="docker"
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/magi

EXPOSE 5555 5556 5557 5558

ENTRYPOINT ["/entrypoint.sh"]
