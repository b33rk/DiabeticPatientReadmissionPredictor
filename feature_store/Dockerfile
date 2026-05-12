FROM python:3.11-slim

WORKDIR /feature_store

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    feast[redis,postgres]==0.40.0 \
    pandas==2.2.0 \
    pyarrow==15.0.0

COPY feature_repo/ ./feature_repo/

# Run feast apply then materialize
# Called as a one-off job: docker-compose --profile feast run --rm feast
CMD ["sh", "-c", \
     "cd feature_repo && feast apply && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)"]
