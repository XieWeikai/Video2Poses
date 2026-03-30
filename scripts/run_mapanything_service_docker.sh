#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="video2poses-mapanything-service:latest"
CONTAINER_NAME="video2poses-mapanything-service"
HOST_PORT="18080"
HOST_ADDR="0.0.0.0"
GPU_SPEC="all"
MODEL_DIR=""
CONFIG_PATH=""
TORCH_CACHE_DIR="${HOME}/.cache/torch"
EXTRA_MOUNTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --host-port)
      HOST_PORT="$2"
      shift 2
      ;;
    --host-addr)
      HOST_ADDR="$2"
      shift 2
      ;;
    --gpus)
      GPU_SPEC="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --torch-cache-dir)
      TORCH_CACHE_DIR="$2"
      shift 2
      ;;
    --mount)
      EXTRA_MOUNTS+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_DIR}" ]]; then
  echo "--model-dir is required" >&2
  exit 1
fi

if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="$(cd "$(dirname "$0")/.." && pwd)/configs/service/mapanything_infer_service.yaml"
fi

DOCKER_ARGS=(
  --rm
  -d
  --name "${CONTAINER_NAME}"
  --gpus "${GPU_SPEC}"
  -p "${HOST_PORT}:18080"
  -e MAPANYTHING_SERVICE_HOST="0.0.0.0"
  -e MAPANYTHING_SERVICE_PORT="18080"
  -e MAPANYTHING_MODEL_DIR="/models/map-anything"
  -e MAPANYTHING_CONFIG_PATH="/configs/service.yaml"
  -v "${MODEL_DIR}:/models/map-anything:ro"
  -v "${CONFIG_PATH}:/configs/service.yaml:ro"
  -v "${TORCH_CACHE_DIR}:/root/.cache/torch"
)

for mount_spec in "${EXTRA_MOUNTS[@]}"; do
  DOCKER_ARGS+=(-v "${mount_spec}")
done

docker run "${DOCKER_ARGS[@]}" \
  "${IMAGE_TAG}"

echo "Container started: ${CONTAINER_NAME}"
echo "Service URL: http://${HOST_ADDR}:${HOST_PORT}"
