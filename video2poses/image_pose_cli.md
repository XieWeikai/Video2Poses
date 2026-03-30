# Image Pose Client

## Overview

[image_pose_cli.py](./image_pose_cli.py) provides a reusable Python client for
services that expose an image-to-pose inference API compatible with this
repository.

The module is designed for direct import into another project. Its primary API
is the `ImagePoseClient` class, which accepts a service URL and exposes typed
Python methods for:

- querying service health,
- submitting a batch of image paths for inference,
- receiving typed dataclass results instead of raw JSON.

The module also contains a minimal command-line wrapper for manual debugging,
but the importable class is the main interface.

## Public API

### `ImagePoseClient`

Constructor:

```python
ImagePoseClient(
    server_url: str = "http://127.0.0.1:18080",
    *,
    request_timeout_sec: float = 1800.0,
)
```

Parameters:

- `server_url`
  Base URL of the running service.
- `request_timeout_sec`
  Timeout applied to requests sent to the service.

The client does not start or stop the service. A running service is expected at
the provided URL.

### Methods

#### `health() -> ServiceHealth`

Query `GET /health` and return a typed `ServiceHealth` dataclass.

#### `infer(image_paths) -> ImagePoseResult`

Send a sequence of image paths to `POST /infer` and return a typed
`ImagePoseResult`.

#### `infer_dir(images_dir, *, glob_pattern="*", sort=True, limit=None) -> ImagePoseResult`

Collect image paths from a directory and then call `infer(...)`.

#### `collect_images_from_dir(images_dir, *, glob_pattern="*", sort=True, limit=None) -> list[str]`

Static helper for converting a directory into an image-path list.

## Returned Dataclasses

### `ServiceHealth`

```python
ServiceHealth(
    ok: bool,
    devices: tuple[str, ...],
    pending: tuple[int, ...],
)
```

### `CameraIntrinsics`

```python
CameraIntrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
)
```

All four fields are in pixel units and refer to the original input-image
resolution.

### `CoordinateSystemSpec`

```python
CoordinateSystemSpec(
    camera_pose_type: str,
    camera_convention: str,
    axes: str,
)
```

### `PoseEstimate`

```python
PoseEstimate(
    image_path: str,
    cam_quats: tuple[float, float, float, float],
    cam_trans: tuple[float, float, float],
    camera_pose: tuple[tuple[float, float, float, float], ...],
    intrinsics: CameraIntrinsics,
    original_image_size: tuple[int, int],
    processed_image_size: tuple[int, int],
    metric_scaling_factor: float,
)
```

### `ImagePoseResult`

```python
ImagePoseResult(
    device: str,
    coordinate_system: CoordinateSystemSpec,
    views: tuple[PoseEstimate, ...],
)
```

## Error Handling

The client raises `ImagePoseClientError` when the service returns a non-200
response.

Attributes:

- `status_code`
  HTTP status code returned by the service
- `code`
  Service-side structured error code when available
- `payload`
  Full parsed error payload

Typical cases:

- `400`
  Invalid request or missing image path
- `503`
  GPU OOM or temporary service-side resource exhaustion
- `504`
  Service-side timeout

## Python Usage

### Example: explicit image list

```python
from video2poses.image_pose_cli import ImagePoseClient

client = ImagePoseClient("http://127.0.0.1:18080")

result = client.infer([
    "/abs/path/img1.png",
    "/abs/path/img2.png",
])

first_view = result.views[0]
print(first_view.image_path)
print(first_view.intrinsics.fx, first_view.intrinsics.fy)
print(first_view.cam_trans)
```

### Example: directory input

```python
from video2poses.image_pose_cli import ImagePoseClient

client = ImagePoseClient("http://127.0.0.1:18080")

result = client.infer_dir(
    "/abs/path/to/frames",
    glob_pattern="*.png",
    sort=True,
    limit=32,
)

print(len(result.views))
```

### Example: error handling

```python
from video2poses.image_pose_cli import ImagePoseClient, ImagePoseClientError

client = ImagePoseClient("http://127.0.0.1:18080")

try:
    result = client.infer(["/abs/path/img1.png"])
except ImagePoseClientError as exc:
    print(exc.status_code)
    print(exc.code)
    print(exc.payload)
```

## Command-Line Wrapper

The module can also be executed directly:

### Health check

```bash
python service/image_pose_cli.py health
```

### Inference with explicit paths

```bash
python service/image_pose_cli.py infer \
  --images /abs/path/img1.png /abs/path/img2.png
```

### Inference from a directory

```bash
python service/image_pose_cli.py infer \
  --images-dir /abs/path/to/frames \
  --glob '*.png' \
  --output result.json
```

The CLI is intended for manual inspection. For integration into another Python
project, direct import of `ImagePoseClient` is recommended.

## Compatibility

This client assumes the service follows the API documented in
[service/service.md](../service/service.md):

- `GET /health`
- `POST /infer`

The returned fields are interpreted according to that schema.
