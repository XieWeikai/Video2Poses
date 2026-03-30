## Image Pose Service

## Overview

This repository wraps the pinned MapAnything submodule in a service that can be
deployed directly or packaged as a Docker container.

Service implementation:

- [mapanything_service.py](./mapanything_service.py)
- [run_mapanything_service.py](./run_mapanything_service.py)

Pinned third-party dependency:

- [third_party/map-anything](../third_party/map-anything)

Default config template:

- [configs/service/mapanything_infer_service.yaml](../configs/service/mapanything_infer_service.yaml)

## Build the Docker Image

```bash
cd path/to/Video2Poses
docker build \
  -f docker/mapanything-service.Dockerfile \
  -t video2poses-mapanything-service:latest \
  .
```

## Run the Docker Container

```bash
bash scripts/run_mapanything_service_docker.sh \
  --model-dir /data-25T/models/map-anything \
  --config-path /home/ubuntu/xwk/learn/Video2Poses/configs/service/mapanything_infer_service.yaml \
  --host-port 18080 \
  --mount path/to/tmp_frames:path/to/tmp_frames:ro \
  --gpus 'all'
```

The helper script mounts:

- the host model directory
- the selected config file
- the host torch cache directory
- any extra path passed through `--mount`

When using the Dockerized service, every image path sent to `/infer` must be
visible inside the container. The simplest pattern is to mount the image
directory to the same absolute path on the host and in the container.

Container-side environment variables:

- `MAPANYTHING_SERVICE_HOST`
- `MAPANYTHING_SERVICE_PORT`
- `MAPANYTHING_MODEL_DIR`
- `MAPANYTHING_CONFIG_PATH`

These are consumed by [docker/entrypoint.sh](../docker/entrypoint.sh).

## Service Configuration

The YAML file contains two top-level sections:

- `service`
- `model`

### `service`

- `host`
  Service bind address
- `port`
  Service bind port
- `log_level`
  Uvicorn log level
- `request_timeout_sec`
  Timeout for one `/infer` request

### `model`

- `model_dir`
  Directory containing `model.safetensors`
- `devices`
  GPU list such as `cuda0` or `cuda:0`
- `infer`
  Parameters forwarded to `model.infer(...)`

## HTTP API

### `GET /health`

Returns service health and queue state.

Example response:

```json
{
  "ok": true,
  "devices": ["cuda:0", "cuda:1"],
  "pending": [0, 1]
}
```

### `POST /infer`

Request body:

```json
{
  "image_paths": [
    "/abs/path/img1.png",
    "/abs/path/img2.png"
  ]
}
```

Successful response:

```json
{
  "device": "cuda:0",
  "coordinate_system": {
    "camera_pose_type": "cam2world",
    "camera_convention": "OpenCV",
    "axes": "+X right, +Y down, +Z forward"
  },
  "views": [
    {
      "image_path": "/abs/path/img1.png",
      "cam_quats": [0.0, 0.0, 0.0, 1.0],
      "cam_trans": [0.0, 0.0, 0.0],
      "camera_pose": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ],
      "intrinsics": {
        "fx": 1100.0,
        "fy": 1100.0,
        "cx": 640.0,
        "cy": 360.0
      },
      "original_image_size": [1280, 720],
      "processed_image_size": [518, 294],
      "metric_scaling_factor": 7.3
    }
  ]
}
```

## Output Semantics

The returned:

- `camera_pose`
- `cam_quats`
- `cam_trans`

are `cam2world` outputs using the OpenCV convention:

- `+X right`
- `+Y down`
- `+Z forward`

The returned:

- `intrinsics.fx`
- `intrinsics.fy`
- `intrinsics.cx`
- `intrinsics.cy`

have already been mapped back to the original input-image resolution and are in
pixel units.

## Failure Modes

### `400`

Invalid request or missing image path.

### `503`

GPU OOM. The failing request returns `503`, but the worker process stays alive
and later smaller requests can still succeed.

### `504`

Request exceeded `service.request_timeout_sec`.

### `500`

Unexpected internal inference error.

服务内部会：
1. 按 `MapAnything` 实际规则计算原图到推理图的 `resize + center crop`
2. 读取模型输出的处理后分辨率内参
3. 反变换回原图坐标系

因此返回值满足：
- 对应原始输入图尺寸
- 单位是像素

### 位姿

服务返回的：
- `cam_quats`
- `cam_trans`
- `camera_pose`

都对应 `MapAnything` 的 `cam2world` 输出。

坐标约定：
- `+X right`
- `+Y down`
- `+Z forward`

## 5. 各类情况下的行为

### 正常请求

- 返回 `HTTP 200`
- body 为上面的完整结果

### 图片路径不存在

- 返回 `HTTP 400`
- `detail` 是错误文本

示例：

```json
{
  "detail": "Image path not found: /bad/path.png"
}
```

### 请求体不合法

例如：
- 缺少 `image_paths`
- `image_paths` 为空

这类由 FastAPI/Pydantic 校验，返回 `HTTP 422`。

### 请求超时

如果单次请求超过 `service.request_timeout_sec`，返回 `HTTP 504`。

示例：

```json
{
  "detail": "Inference request timed out."
}
```

### CUDA Out Of Memory

如果某次请求在某张卡上触发 OOM：

- 返回 `HTTP 503`
- 返回结构化错误码 `cuda_out_of_memory`
- 服务不会退出
- 对应 worker 会执行显存清理
- 后续更小的请求仍可继续服务

返回示例：

```json
{
  "detail": {
    "code": "cuda_out_of_memory",
    "message": "CUDA out of memory on cuda:0. The request failed, but the service stayed alive."
  }
}
```

这条行为已经做过真实验证：服务在一次 OOM 后，下一次小请求仍然成功返回 `200`。

### 其他推理错误

其他内部异常返回 `HTTP 500`，格式如下：

```json
{
  "detail": {
    "code": "inference_failed",
    "message": "..."
  }
}
```

如果是更外层的运行时错误，则返回：

```json
{
  "detail": {
    "code": "runtime_error",
    "message": "..."
  }
}
```

### 启动失败

如果：
- 模型路径不对
- 权重文件缺失
- CUDA 设备不可用

则 worker 无法 ready，服务启动阶段会直接失败，而不是进入可服务状态。

## 6. 简单调用示例

```bash
curl -X POST http://127.0.0.1:18080/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "image_paths": [
      "/abs/path/img1.png",
      "/abs/path/img2.png"
    ]
  }'
```

健康检查：

```bash
curl http://127.0.0.1:18080/health
```
