#!/usr/bin/env bash
set -euo pipefail

SERVICE_HOST="${MAPANYTHING_SERVICE_HOST:-0.0.0.0}"
SERVICE_PORT="${MAPANYTHING_SERVICE_PORT:-18080}"
MODEL_DIR="${MAPANYTHING_MODEL_DIR:-/models/map-anything}"
CONFIG_PATH="${MAPANYTHING_CONFIG_PATH:-/app/configs/service/mapanything_infer_service.yaml}"

exec /opt/conda/bin/conda run --no-capture-output -n service python /app/service/run_mapanything_service.py \
  --config "${CONFIG_PATH}" \
  --host "${SERVICE_HOST}" \
  --port "${SERVICE_PORT}" \
  --model-dir "${MODEL_DIR}"
