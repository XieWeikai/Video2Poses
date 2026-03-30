#!/usr/bin/env python3
"""Reusable Python client for image-to-pose services.

This module provides a small, typed wrapper around the local image-pose HTTP
service implemented in this repository. The wrapper is intentionally generic:

- the class name does not mention MapAnything,
- the constructor only needs a service URL,
- the return type is a set of dataclasses instead of raw JSON.

The intended usage is from another Python project:

    from video2poses.image_pose_cli import ImagePoseClient

    client = ImagePoseClient("http://127.0.0.1:18080")
    result = client.infer([
        "/abs/path/img1.png",
        "/abs/path/img2.png",
    ])

    first_view = result.views[0]
    print(first_view.intrinsics.fx, first_view.intrinsics.fy)
    print(first_view.cam_trans)

The module also exposes a very small CLI wrapper for quick manual checks, but
the primary API is the `ImagePoseClient` class.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_SERVER_URL = "http://127.0.0.1:18080"
DEFAULT_REQUEST_TIMEOUT_SEC = 1800.0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in the original image coordinate system.

    All four values are in pixel units and correspond to the original input
    image resolution, not the internal resized/cropped inference resolution.
    """

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CoordinateSystemSpec:
    """Description of the pose convention used by the service output."""

    camera_pose_type: str
    camera_convention: str
    axes: str


@dataclass(frozen=True)
class PoseEstimate:
    """Per-image camera prediction returned by the service."""

    image_path: str
    cam_quats: tuple[float, float, float, float]
    cam_trans: tuple[float, float, float]
    camera_pose: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    intrinsics: CameraIntrinsics
    original_image_size: tuple[int, int]
    processed_image_size: tuple[int, int]
    metric_scaling_factor: float


@dataclass(frozen=True)
class ImagePoseResult:
    """Structured inference result for a batch of images."""

    device: str
    coordinate_system: CoordinateSystemSpec
    views: tuple[PoseEstimate, ...]


@dataclass(frozen=True)
class ServiceHealth:
    """Structured `/health` payload returned by the service."""

    ok: bool
    devices: tuple[str, ...]
    pending: tuple[int, ...]


class ImagePoseClientError(RuntimeError):
    """Raised when the remote service returns a non-success response."""

    def __init__(self, status_code: int, code: str | None, message: str, payload: dict[str, Any]):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.payload = payload


def _normalize_server_url(server_url: str) -> str:
    """Return a normalized base URL without a trailing slash."""

    return server_url.rstrip("/")


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_sec: float,
) -> tuple[int, dict[str, Any]]:
    """Send an HTTP request and return `(status_code, parsed_json_body)`.

    The service uses JSON for both successful and error responses, so the
    client always returns parsed JSON regardless of the status code.
    """

    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload_obj = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload_obj = {"raw_body": body}
        return exc.code, payload_obj


def _normalize_image_paths(image_paths: Iterable[str | Path]) -> list[str]:
    """Resolve, validate, and deduplicate image paths."""

    normalized: list[str] = []
    seen: set[str] = set()
    for item in image_paths:
        resolved = str(Path(item).expanduser().resolve())
        if resolved in seen:
            continue
        if not Path(resolved).exists():
            raise FileNotFoundError(f"Image path not found: {resolved}")
        seen.add(resolved)
        normalized.append(resolved)

    if not normalized:
        raise ValueError("image_paths must not be empty")
    return normalized


def _parse_coordinate_system(payload: dict[str, Any]) -> CoordinateSystemSpec:
    """Convert raw coordinate-system JSON into a dataclass."""

    return CoordinateSystemSpec(
        camera_pose_type=str(payload["camera_pose_type"]),
        camera_convention=str(payload["camera_convention"]),
        axes=str(payload["axes"]),
    )


def _parse_intrinsics(payload: dict[str, Any]) -> CameraIntrinsics:
    """Convert raw intrinsics JSON into a dataclass."""

    return CameraIntrinsics(
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
    )


def _parse_pose_estimate(payload: dict[str, Any]) -> PoseEstimate:
    """Convert one raw view prediction into a typed dataclass."""

    camera_pose_rows = tuple(
        tuple(float(value) for value in row) for row in payload["camera_pose"]
    )
    return PoseEstimate(
        image_path=str(payload["image_path"]),
        cam_quats=tuple(float(value) for value in payload["cam_quats"]),
        cam_trans=tuple(float(value) for value in payload["cam_trans"]),
        camera_pose=camera_pose_rows,  # type: ignore[arg-type]
        intrinsics=_parse_intrinsics(payload["intrinsics"]),
        original_image_size=tuple(int(value) for value in payload["original_image_size"]),
        processed_image_size=tuple(int(value) for value in payload["processed_image_size"]),
        metric_scaling_factor=float(payload["metric_scaling_factor"]),
    )


def _parse_result(payload: dict[str, Any]) -> ImagePoseResult:
    """Convert the raw inference JSON payload into typed dataclasses."""

    return ImagePoseResult(
        device=str(payload["device"]),
        coordinate_system=_parse_coordinate_system(payload["coordinate_system"]),
        views=tuple(_parse_pose_estimate(view) for view in payload["views"]),
    )


def _parse_health(payload: dict[str, Any]) -> ServiceHealth:
    """Convert the raw `/health` JSON payload into a dataclass."""

    return ServiceHealth(
        ok=bool(payload["ok"]),
        devices=tuple(str(device) for device in payload["devices"]),
        pending=tuple(int(value) for value in payload["pending"]),
    )


class ImagePoseClient:
    """Typed Python client for a service that maps images to camera poses.

    The class expects a service that exposes the API documented by
    `video2poses/service.md` in this repository:

    - `GET /health`
    - `POST /infer`

    It does not start or stop the service. Service lifecycle is intentionally
    left outside of this helper so that the client remains simple to reuse.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        *,
        request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> None:
        """Create a client bound to one running service URL."""

        self.server_url = _normalize_server_url(server_url)
        self.request_timeout_sec = float(request_timeout_sec)

    @staticmethod
    def collect_images_from_dir(
        images_dir: str | Path,
        *,
        glob_pattern: str = "*",
        sort: bool = True,
        limit: int | None = None,
    ) -> list[str]:
        """Collect image paths from a directory for convenience.

        This helper is optional. Callers can also directly provide an explicit
        list of image paths to `infer(...)`.
        """

        root = Path(images_dir).expanduser().resolve()
        matched = [
            path
            for path in root.glob(glob_pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if sort:
            matched = sorted(matched)
        if limit is not None:
            matched = matched[:limit]
        return [str(path) for path in matched]

    def health(self) -> ServiceHealth:
        """Return the typed health payload for the remote service."""

        status_code, payload = _request_json(
            "GET",
            self.server_url + "/health",
            timeout_sec=min(10.0, self.request_timeout_sec),
        )
        if status_code != 200:
            raise ImagePoseClientError(
                status_code=status_code,
                code=None,
                message=f"Health check failed with status {status_code}",
                payload=payload,
            )
        return _parse_health(payload)

    def infer(self, image_paths: Sequence[str | Path]) -> ImagePoseResult:
        """Infer camera poses and intrinsics for the given images.

        Args:
            image_paths:
                Sequence of image paths to send to the service.

        Returns:
            `ImagePoseResult`, which contains typed per-image predictions.

        Raises:
            FileNotFoundError:
                If one of the input paths does not exist.
            ValueError:
                If the path sequence is empty.
            ImagePoseClientError:
                If the service returns a non-200 response such as 400, 503, or
                504.
        """

        normalized_paths = _normalize_image_paths(image_paths)
        status_code, payload = _request_json(
            "POST",
            self.server_url + "/infer",
            payload={"image_paths": normalized_paths},
            timeout_sec=self.request_timeout_sec,
        )
        if status_code != 200:
            detail = payload.get("detail", {})
            if isinstance(detail, dict):
                code = detail.get("code")
                message = str(detail.get("message", f"Request failed with status {status_code}"))
            else:
                code = None
                message = str(detail)
            raise ImagePoseClientError(
                status_code=status_code,
                code=code,
                message=message,
                payload=payload,
            )
        return _parse_result(payload)

    def infer_dir(
        self,
        images_dir: str | Path,
        *,
        glob_pattern: str = "*",
        sort: bool = True,
        limit: int | None = None,
    ) -> ImagePoseResult:
        """Collect images from a directory and run inference."""

        image_paths = self.collect_images_from_dir(
            images_dir,
            glob_pattern=glob_pattern,
            sort=sort,
            limit=limit,
        )
        return self.infer(image_paths)


def _build_parser() -> argparse.ArgumentParser:
    """Build a minimal optional CLI wrapper around the typed client."""

    parser = argparse.ArgumentParser(description="Typed client for image-pose services.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--request-timeout-sec", type=float, default=DEFAULT_REQUEST_TIMEOUT_SEC)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--images", nargs="+")
    infer_parser.add_argument("--images-dir")
    infer_parser.add_argument("--glob", default="*")
    infer_parser.add_argument("--no-sort", action="store_true")
    infer_parser.add_argument("--limit", type=int)
    infer_parser.add_argument("--output")

    return parser


def main() -> int:
    """Optional thin CLI entrypoint for manual debugging."""

    parser = _build_parser()
    args = parser.parse_args()
    client = ImagePoseClient(
        server_url=args.server_url,
        request_timeout_sec=args.request_timeout_sec,
    )

    try:
        if args.command == "health":
            payload = asdict(client.health())
        elif args.command == "infer":
            if args.images:
                payload = asdict(client.infer(args.images))
            elif args.images_dir:
                payload = asdict(
                    client.infer_dir(
                        args.images_dir,
                        glob_pattern=args.glob,
                        sort=not args.no_sort,
                        limit=args.limit,
                    )
                )
            else:
                parser.error("infer requires --images or --images-dir")
                return 2
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except ImagePoseClientError as exc:
        print(
            json.dumps(
                {
                    "status_code": exc.status_code,
                    "code": exc.code,
                    "message": str(exc),
                    "payload": exc.payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if getattr(args, "output", None):
        Path(args.output).expanduser().write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
