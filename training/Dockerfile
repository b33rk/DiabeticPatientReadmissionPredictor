# CUDA 12.1 runtime matches the P100's driver on the server node.
# Use "runtime" not "devel" — we don't compile CUDA code, XGBoost ships pre-built.
FROM python:3.11-slim
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         python3.11 \
#         python3.11-dev \
#         python3-pip \
#         libpq-dev \
#         build-essential \
#     && ln -sf python3.11 /usr/bin/python3 \
#     && ln -sf python3 /usr/bin/python \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The training script is the entrypoint; override CMD for different run modes:
#   docker-compose run training python src/train.py --config configs/train_config.yaml
ENTRYPOINT ["python", "src/train.py"]
