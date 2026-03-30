from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .image_pose_cli import CameraIntrinsics, CoordinateSystemSpec


Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class VideoPoseConfig:
    input_dir: Path
    server_url: str
    sample_fps: float
    initial_max_inference_frames: int
    min_inference_frames: int
    initial_max_concurrent: int
    max_video_workers: int
    max_retries: int
    request_timeout_sec: float
    output_dir: Path | None = None
    keep_temp: bool = False
    temp_root_dir: Path | None = None
    max_inference_frames_cap: int | None = None
    max_concurrency_cap: int | None = None
    success_window: int = 3
    base_backoff_sec: float = 1.0
    max_backoff_sec: float = 30.0
    growth_max_latency_sec: float = 15.0
    growth_max_latency_per_frame_sec: float = 0.75
    degrade_latency_sec: float = 45.0
    degrade_latency_per_frame_sec: float = 1.5

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")
        if self.initial_max_inference_frames < 1:
            raise ValueError("initial_max_inference_frames must be >= 1")
        if self.min_inference_frames < 1:
            raise ValueError("min_inference_frames must be >= 1")
        if self.min_inference_frames > self.initial_max_inference_frames:
            raise ValueError("min_inference_frames must be <= initial_max_inference_frames")
        if self.initial_max_concurrent < 1:
            raise ValueError("initial_max_concurrent must be >= 1")
        if self.max_video_workers < 1:
            raise ValueError("max_video_workers must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be > 0")
        if self.success_window < 1:
            raise ValueError("success_window must be >= 1")
        if self.base_backoff_sec < 0:
            raise ValueError("base_backoff_sec must be >= 0")
        if self.max_backoff_sec < self.base_backoff_sec:
            raise ValueError("max_backoff_sec must be >= base_backoff_sec")
        if self.growth_max_latency_sec <= 0:
            raise ValueError("growth_max_latency_sec must be > 0")
        if self.growth_max_latency_per_frame_sec <= 0:
            raise ValueError("growth_max_latency_per_frame_sec must be > 0")
        if self.degrade_latency_sec <= 0:
            raise ValueError("degrade_latency_sec must be > 0")
        if self.degrade_latency_per_frame_sec <= 0:
            raise ValueError("degrade_latency_per_frame_sec must be > 0")
        if self.degrade_latency_sec < self.growth_max_latency_sec:
            raise ValueError("degrade_latency_sec must be >= growth_max_latency_sec")
        if self.degrade_latency_per_frame_sec < self.growth_max_latency_per_frame_sec:
            raise ValueError(
                "degrade_latency_per_frame_sec must be >= growth_max_latency_per_frame_sec"
            )


@dataclass(frozen=True)
class VideoJob:
    video_id: str
    video_path: Path
    output_path: Path


@dataclass(frozen=True)
class VideoMetadata:
    source_video_fps: float
    duration_sec: float
    total_frames: int
    width: int
    height: int


@dataclass(frozen=True)
class FrameRecord:
    sample_index: int
    source_frame_index: int
    timestamp_sec: float
    image_path: Path


@dataclass(frozen=True)
class InferenceChunk:
    video_id: str
    chunk_index: int
    start_sample_index: int
    frame_records: tuple[FrameRecord, ...]
    has_overlap_anchor: bool

    @property
    def image_paths(self) -> list[str]:
        return [str(record.image_path) for record in self.frame_records]

    @property
    def unique_frame_records(self) -> tuple[FrameRecord, ...]:
        if self.has_overlap_anchor and len(self.frame_records) > 1:
            return self.frame_records[1:]
        return self.frame_records

    @property
    def anchor_frame(self) -> FrameRecord | None:
        if self.has_overlap_anchor and self.frame_records:
            return self.frame_records[0]
        return None


@dataclass(frozen=True)
class FramePoseRecord:
    sample_index: int
    source_frame_index: int
    timestamp_sec: float
    intrinsics: CameraIntrinsics
    original_image_size: tuple[int, int]
    processed_image_size: tuple[int, int]
    metric_scaling_factor: float
    camera_pose: Matrix4
    cam_quats: tuple[float, float, float, float]
    cam_trans: tuple[float, float, float]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "source_frame_index": self.source_frame_index,
            "timestamp_sec": self.timestamp_sec,
            "image_size": [self.original_image_size[0], self.original_image_size[1]],
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
            },
            "pose": {
                "camera_pose": [list(row) for row in self.camera_pose],
                "cam_quats": list(self.cam_quats),
                "cam_trans": list(self.cam_trans),
            },
        }


@dataclass(frozen=True)
class ChunkInferenceResult:
    chunk: InferenceChunk
    aligned_frames: tuple[FramePoseRecord, ...]
    anchor_global_pose: Matrix4


@dataclass(frozen=True)
class VideoCameraInfo:
    schema_version: str
    video_id: str
    source_video: str
    source_video_fps: float
    sample_fps: float
    num_sampled_frames: int
    coordinate_system: CoordinateSystemSpec
    frames: tuple[FramePoseRecord, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "source_video": self.source_video,
            "source_video_fps": self.source_video_fps,
            "sample_fps": self.sample_fps,
            "num_sampled_frames": self.num_sampled_frames,
            "coordinate_system": {
                "camera_pose_type": self.coordinate_system.camera_pose_type,
                "camera_convention": self.coordinate_system.camera_convention,
                "axes": self.coordinate_system.axes,
            },
            "frames": [frame.to_json_dict() for frame in self.frames],
        }


@dataclass(frozen=True)
class BatchFailure:
    video_id: str
    video_path: str
    error: str


@dataclass(frozen=True)
class BatchSummary:
    input_dir: str
    output_dir: str
    total_videos: int
    succeeded: int
    failed: int
    output_files: tuple[str, ...] = field(default_factory=tuple)
    failures: tuple[BatchFailure, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "total_videos": self.total_videos,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "output_files": list(self.output_files),
            "failures": [
                {
                    "video_id": failure.video_id,
                    "video_path": failure.video_path,
                    "error": failure.error,
                }
                for failure in self.failures
            ],
        }
