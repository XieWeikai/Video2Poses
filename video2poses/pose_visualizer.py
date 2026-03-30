from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

from .image_pose_cli import CameraIntrinsics, CoordinateSystemSpec
from .video_io import ensure_ffmpeg_available


matplotlib.use("Agg")
from matplotlib import pyplot as plt


LOGGER = logging.getLogger(__name__)
SUPPORTED_CAMERA_CONVENTIONS = {"OpenCV", "OpenGL"}
CONVENTION_SWAP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class CameraInfoFrame:
    sample_index: int
    source_frame_index: int
    timestamp_sec: float
    image_size: tuple[int, int]
    intrinsics: CameraIntrinsics
    camera_pose: np.ndarray
    cam_quats: tuple[float, float, float, float]
    cam_trans: tuple[float, float, float]


@dataclass(frozen=True)
class CameraInfoDocument:
    schema_version: str
    video_id: str
    source_video: Path
    source_video_fps: float
    sample_fps: float
    num_sampled_frames: int
    coordinate_system: CoordinateSystemSpec
    frames: tuple[CameraInfoFrame, ...]

    @property
    def image_size(self) -> tuple[int, int]:
        if not self.frames:
            raise ValueError("camera info does not contain any frames")
        return self.frames[0].image_size


@dataclass(frozen=True)
class PoseVisualizationConfig:
    camera_json_path: Path
    output_video_path: Path | None = None
    view_mode: str = "3d"
    visualization_convention: str | None = None
    output_fps: float | None = None
    frustum_scale: float = 1.0
    history_window_sec: float = 10.0
    max_history_frustums: int = 80
    keep_temp: bool = False
    temp_root_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.view_mode not in {"2d", "3d"}:
            raise ValueError("view_mode must be one of: 2d, 3d")
        if self.visualization_convention is not None:
            _validate_camera_convention(self.visualization_convention)
        if self.output_fps is not None and self.output_fps <= 0:
            raise ValueError("output_fps must be > 0")
        if self.frustum_scale <= 0:
            raise ValueError("frustum_scale must be > 0")
        if self.history_window_sec <= 0:
            raise ValueError("history_window_sec must be > 0")
        if self.max_history_frustums < 1:
            raise ValueError("max_history_frustums must be >= 1")


@dataclass(frozen=True)
class VisualizationState:
    target_convention: str
    poses: tuple[np.ndarray, ...]
    centers: np.ndarray
    frusta: tuple[np.ndarray, ...]
    display_centers_3d: np.ndarray
    display_frusta_3d: tuple[np.ndarray, ...]
    history_start_indices: tuple[int, ...]
    window_bounds_2d: tuple[tuple[float, float, float, float], ...]
    window_bounds_3d: tuple[tuple[np.ndarray, np.ndarray], ...]
    frustum_depth_m: float


def _validate_camera_convention(camera_convention: str) -> None:
    if camera_convention not in SUPPORTED_CAMERA_CONVENTIONS:
        raise ValueError(
            "camera_convention must be one of: " + ", ".join(sorted(SUPPORTED_CAMERA_CONVENTIONS))
        )


def build_default_visualization_output_path(camera_json_path: Path, view_mode: str) -> Path:
    resolved_path = camera_json_path.expanduser().resolve()
    stem = resolved_path.stem
    if stem.endswith("-camera"):
        stem = stem[: -len("-camera")]
    return resolved_path.with_name(f"{stem}-{view_mode}-visualization.mp4")


def load_camera_info(camera_json_path: str | Path) -> CameraInfoDocument:
    path = Path(camera_json_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    coordinate_payload = payload["coordinate_system"]
    coordinate_system = CoordinateSystemSpec(
        camera_pose_type=str(coordinate_payload["camera_pose_type"]),
        camera_convention=str(coordinate_payload["camera_convention"]),
        axes=str(coordinate_payload["axes"]),
    )
    _validate_camera_convention(coordinate_system.camera_convention)

    frames: list[CameraInfoFrame] = []
    expected_image_size: tuple[int, int] | None = None
    for index, frame_payload in enumerate(payload["frames"]):
        pose_payload = frame_payload["pose"]
        camera_pose = np.asarray(pose_payload["camera_pose"], dtype=np.float64)
        if camera_pose.shape != (4, 4):
            raise ValueError(f"frame {index} camera_pose must be a 4x4 matrix")
        if int(frame_payload["sample_index"]) != index:
            raise ValueError("frames must be ordered by contiguous sample_index starting from 0")

        image_size = tuple(int(value) for value in frame_payload["image_size"])
        if len(image_size) != 2:
            raise ValueError(f"frame {index} image_size must contain exactly two integers")
        if expected_image_size is None:
            expected_image_size = (image_size[0], image_size[1])
        elif expected_image_size != (image_size[0], image_size[1]):
            raise ValueError("all frames must share the same image_size for video rendering")

        intrinsics_payload = frame_payload["intrinsics"]
        intrinsics = CameraIntrinsics(
            fx=float(intrinsics_payload["fx"]),
            fy=float(intrinsics_payload["fy"]),
            cx=float(intrinsics_payload["cx"]),
            cy=float(intrinsics_payload["cy"]),
        )
        frames.append(
            CameraInfoFrame(
                sample_index=int(frame_payload["sample_index"]),
                source_frame_index=int(frame_payload["source_frame_index"]),
                timestamp_sec=float(frame_payload["timestamp_sec"]),
                image_size=(image_size[0], image_size[1]),
                intrinsics=intrinsics,
                camera_pose=camera_pose,
                cam_quats=tuple(float(value) for value in pose_payload["cam_quats"]),
                cam_trans=tuple(float(value) for value in pose_payload["cam_trans"]),
            )
        )

    document = CameraInfoDocument(
        schema_version=str(payload["schema_version"]),
        video_id=str(payload["video_id"]),
        source_video=Path(payload["source_video"]).expanduser().resolve(),
        source_video_fps=float(payload["source_video_fps"]),
        sample_fps=float(payload["sample_fps"]),
        num_sampled_frames=int(payload["num_sampled_frames"]),
        coordinate_system=coordinate_system,
        frames=tuple(frames),
    )
    if document.num_sampled_frames != len(document.frames):
        raise ValueError(
            "num_sampled_frames does not match frame count: "
            f"{document.num_sampled_frames} != {len(document.frames)}"
        )
    if not document.frames:
        raise ValueError("camera info file does not contain any frames")
    return document


def convert_camera_pose_convention(
    camera_pose: np.ndarray,
    *,
    from_convention: str,
    to_convention: str,
) -> np.ndarray:
    _validate_camera_convention(from_convention)
    _validate_camera_convention(to_convention)
    pose = np.asarray(camera_pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera_pose must be a 4x4 matrix")
    if from_convention == to_convention:
        return pose.copy()
    return CONVENTION_SWAP @ pose @ CONVENTION_SWAP


def _build_temp_dir(prefix: str, temp_root_dir: Path | None) -> Path:
    root = temp_root_dir.expanduser().resolve() if temp_root_dir is not None else None
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(root) if root else None))


def _cleanup_temp_dir(temp_dir: Path, keep_temp: bool) -> None:
    if keep_temp:
        return
    shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_sampled_video_frames(camera_info: CameraInfoDocument, output_dir: Path) -> tuple[Path, ...]:
    if not camera_info.source_video.exists():
        raise FileNotFoundError(f"source video not found: {camera_info.source_video}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%06d.png"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(camera_info.source_video),
            "-vf",
            f"fps={camera_info.sample_fps:g}",
            "-frames:v",
            str(camera_info.num_sampled_frames),
            "-start_number",
            "0",
            str(output_pattern),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "ffmpeg failed"
        raise RuntimeError(f"failed to extract sampled frames: {stderr}")

    frames = tuple(sorted(output_dir.glob("frame_*.png")))
    if len(frames) != camera_info.num_sampled_frames:
        raise RuntimeError(
            "sampled frame count mismatch: "
            f"expected {camera_info.num_sampled_frames}, got {len(frames)}"
        )
    return frames


def _compose_visualization_video(
    left_frames_dir: Path,
    right_frames_dir: Path,
    output_video_path: Path,
    *,
    fps: float,
) -> None:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    left_pattern = left_frames_dir / "frame_%06d.png"
    right_pattern = right_frames_dir / "frame_%06d.png"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(left_pattern),
            "-framerate",
            f"{fps:g}",
            "-i",
            str(right_pattern),
            "-filter_complex",
            "hstack=inputs=2,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "ffmpeg failed"
        raise RuntimeError(f"failed to compose visualization video: {stderr}")


def _compute_frustum_depth_m(centers: np.ndarray, frustum_scale: float) -> float:
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) <= 1:
        return 0.15 * frustum_scale

    step_sizes = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive_steps = step_sizes[step_sizes > 1e-8]
    scene_span = np.max(np.ptp(centers, axis=0)) if len(centers) > 0 else 0.0
    typical_step = float(np.median(positive_steps)) if positive_steps.size else max(scene_span, 0.1)
    lower = max(0.03, scene_span * 0.01)
    upper = max(lower, scene_span * 0.08, 0.25)
    depth = np.clip(typical_step * 0.45, lower, upper)
    return float(depth * frustum_scale)


def _project_top_down(points: np.ndarray, camera_convention: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    forward = points[:, 2] if camera_convention == "OpenCV" else -points[:, 2]
    return np.column_stack([points[:, 0], forward])


def _project_display_3d(points: np.ndarray, camera_convention: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if camera_convention == "OpenCV":
        return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])
    return np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])


def _build_local_frustum(frame: CameraInfoFrame, camera_convention: str, depth_m: float) -> np.ndarray:
    width, height = frame.image_size
    pixel_corners = np.asarray(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)],
        ],
        dtype=np.float64,
    )
    x_dirs = (pixel_corners[:, 0] - frame.intrinsics.cx) / frame.intrinsics.fx
    y_dirs = (pixel_corners[:, 1] - frame.intrinsics.cy) / frame.intrinsics.fy
    rays = np.column_stack([x_dirs, y_dirs, np.ones(4, dtype=np.float64)])
    if camera_convention == "OpenGL":
        rays = (CONVENTION_SWAP[:3, :3] @ rays.T).T
    rays *= depth_m / np.abs(rays[:, 2:3])
    return np.vstack([np.zeros((1, 3), dtype=np.float64), rays])


def _transform_points(camera_pose: np.ndarray, local_points: np.ndarray) -> np.ndarray:
    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]
    return (rotation @ local_points.T).T + translation


def _history_window_size_frames(sample_fps: float, history_window_sec: float) -> int:
    return max(1, int(np.ceil(sample_fps * history_window_sec)))


def _compute_history_start_indices(num_frames: int, window_size_frames: int) -> tuple[int, ...]:
    return tuple(max(0, frame_index - window_size_frames + 1) for frame_index in range(num_frames))


def _compute_window_bounds_3d(
    frusta: Sequence[np.ndarray],
    history_start_indices: Sequence[int],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_index, start_index in enumerate(history_start_indices):
        window_points = np.concatenate(frusta[start_index : frame_index + 1], axis=0)
        bounds.append((np.min(window_points, axis=0), np.max(window_points, axis=0)))
    return tuple(bounds)


def _compute_window_bounds_2d(
    frusta: Sequence[np.ndarray],
    *,
    camera_convention: str,
    history_start_indices: Sequence[int],
) -> tuple[tuple[float, float, float, float], ...]:
    bounds: list[tuple[float, float, float, float]] = []
    for frame_index, start_index in enumerate(history_start_indices):
        projected_windows = [
            _project_top_down(frustum, camera_convention)
            for frustum in frusta[start_index : frame_index + 1]
        ]
        window_points = np.concatenate(projected_windows, axis=0)
        current_min = np.min(window_points, axis=0)
        current_max = np.max(window_points, axis=0)
        bounds.append((float(current_min[0]), float(current_max[0]), float(current_min[1]), float(current_max[1])))
    return tuple(bounds)


def build_visualization_state(
    camera_info: CameraInfoDocument,
    *,
    visualization_convention: str | None,
    frustum_scale: float,
    history_window_sec: float,
) -> VisualizationState:
    target_convention = visualization_convention or camera_info.coordinate_system.camera_convention
    _validate_camera_convention(target_convention)

    poses = tuple(
        convert_camera_pose_convention(
            frame.camera_pose,
            from_convention=camera_info.coordinate_system.camera_convention,
            to_convention=target_convention,
        )
        for frame in camera_info.frames
    )
    centers = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    frustum_depth_m = _compute_frustum_depth_m(centers, frustum_scale)
    frusta = tuple(
        _transform_points(pose, _build_local_frustum(frame, target_convention, frustum_depth_m))
        for frame, pose in zip(camera_info.frames, poses, strict=True)
    )
    display_centers_3d = _project_display_3d(centers, target_convention)
    display_frusta_3d = tuple(_project_display_3d(frustum, target_convention) for frustum in frusta)
    history_start_indices = _compute_history_start_indices(
        len(camera_info.frames),
        _history_window_size_frames(camera_info.sample_fps, history_window_sec),
    )
    return VisualizationState(
        target_convention=target_convention,
        poses=poses,
        centers=centers,
        frusta=frusta,
        display_centers_3d=display_centers_3d,
        display_frusta_3d=display_frusta_3d,
        history_start_indices=history_start_indices,
        window_bounds_2d=_compute_window_bounds_2d(
            frusta,
            camera_convention=target_convention,
            history_start_indices=history_start_indices,
        ),
        window_bounds_3d=_compute_window_bounds_3d(
            display_frusta_3d,
            history_start_indices=history_start_indices,
        ),
        frustum_depth_m=frustum_depth_m,
    )


def _history_indices(start_index: int, end_index: int, max_history_frustums: int) -> list[int]:
    if end_index <= start_index:
        return []
    if end_index - start_index <= max_history_frustums:
        return list(range(start_index, end_index))
    sampled = np.linspace(start_index, end_index - 1, num=max_history_frustums, dtype=int)
    return sorted({int(value) for value in sampled})


def _expand_axis_limits_2d(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = bounds
    width = max(x_max - x_min, 1e-3)
    height = max(y_max - y_min, 1e-3)
    span = max(width, height, 0.5)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    margin = span * 0.18 + 0.05
    half = 0.5 * span + margin
    return (x_center - half, x_center + half, y_center - half, y_center + half)


def _expand_axis_limits_3d(bounds: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    min_values, max_values = bounds
    span = np.maximum(max_values - min_values, 1e-3)
    max_span = max(float(np.max(span)), 0.5)
    margin = max_span * 0.18 + 0.05
    center = 0.5 * (min_values + max_values)
    half = 0.5 * max_span + margin
    return center - half, center + half


def _draw_frustum_2d_for_convention(
    ax,
    frustum: np.ndarray,
    *,
    camera_convention: str,
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    projected = _project_top_down(frustum, camera_convention)
    left_corner = projected[1]
    right_corner = projected[2]
    camera_center = projected[0]
    polygon = np.vstack([camera_center, left_corner, right_corner, camera_center])
    ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha * 0.08)


def _draw_current_axes_3d(
    ax,
    pose: np.ndarray,
    *,
    camera_convention: str,
    depth_m: float,
) -> None:
    center = pose[:3, 3]
    axes = pose[:3, :3] @ (np.eye(3, dtype=np.float64) * depth_m * 0.45)
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for axis_vector, color in zip(axes.T, colors, strict=True):
        points = np.vstack([center, center + axis_vector])
        display_points = _project_display_3d(points, camera_convention)
        ax.plot(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            color=color,
            linewidth=2.0,
            alpha=0.95,
        )


def _draw_frustum_3d(ax, frustum: np.ndarray, color: str, alpha: float, linewidth: float) -> None:
    center = frustum[0]
    corners = frustum[1:]
    for corner in corners:
        points = np.vstack([center, corner])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth, alpha=alpha)
    rectangle = np.vstack([corners, corners[0]])
    ax.plot(
        rectangle[:, 0],
        rectangle[:, 1],
        rectangle[:, 2],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )


def _render_2d_frame(
    *,
    ax,
    frame_index: int,
    camera_info: CameraInfoDocument,
    state: VisualizationState,
    max_history_frustums: int,
) -> None:
    x_min, x_max, y_min, y_max = _expand_axis_limits_2d(state.window_bounds_2d[frame_index])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.set_facecolor("#fcfcfc")

    history_start_index = state.history_start_indices[frame_index]
    history = state.centers[history_start_index : frame_index + 1]
    history_top_down = _project_top_down(history, state.target_convention)
    ax.plot(
        history_top_down[:, 0],
        history_top_down[:, 1],
        color="#4a4a4a",
        linewidth=1.8,
        alpha=0.75,
        label="camera path",
    )

    for history_index in _history_indices(history_start_index, frame_index, max_history_frustums):
        _draw_frustum_2d_for_convention(
            ax,
            state.frusta[history_index],
            camera_convention=state.target_convention,
            color="#7aa6ff",
            alpha=0.22,
            linewidth=1.1,
        )

    _draw_frustum_2d_for_convention(
        ax,
        state.frusta[frame_index],
        camera_convention=state.target_convention,
        color="#e4572e",
        alpha=0.95,
        linewidth=2.2,
    )
    ax.scatter(
        history_top_down[-1, 0],
        history_top_down[-1, 1],
        color="#e4572e",
        s=48,
        zorder=5,
    )

    forward_label = "Forward Z (m)" if state.target_convention == "OpenCV" else "Forward -Z (m)"
    ax.set_xlabel("Right X (m)")
    ax.set_ylabel(forward_label)

    frame = camera_info.frames[frame_index]
    current_center = state.centers[frame_index]
    current_forward = current_center[2] if state.target_convention == "OpenCV" else -current_center[2]
    ax.set_title(
        f"2D Frustum View ({state.target_convention})\n"
        f"sample={frame.sample_index}  t={frame.timestamp_sec:.3f}s",
        fontsize=12,
    )
    ax.text(
        0.02,
        0.98,
        (
            f"x={current_center[0]:.3f} m\n"
            f"forward={current_forward:.3f} m\n"
            f"source_frame={frame.source_frame_index}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.35"},
    )


def _render_3d_frame(
    *,
    ax,
    frame_index: int,
    camera_info: CameraInfoDocument,
    state: VisualizationState,
    max_history_frustums: int,
) -> None:
    min_values, max_values = _expand_axis_limits_3d(state.window_bounds_3d[frame_index])
    ax.set_xlim(min_values[0], max_values[0])
    ax.set_ylim(min_values[1], max_values[1])
    ax.set_zlim(min_values[2], max_values[2])
    ax.set_box_aspect(tuple(max_values - min_values))
    ax.view_init(elev=20, azim=-58)

    history_start_index = state.history_start_indices[frame_index]
    history = state.display_centers_3d[history_start_index : frame_index + 1]
    ax.plot(
        history[:, 0],
        history[:, 1],
        history[:, 2],
        color="#4a4a4a",
        linewidth=1.6,
        alpha=0.75,
    )

    for history_index in _history_indices(history_start_index, frame_index, max_history_frustums):
        _draw_frustum_3d(
            ax,
            state.display_frusta_3d[history_index],
            color="#7aa6ff",
            alpha=0.18,
            linewidth=0.9,
        )

    _draw_frustum_3d(
        ax,
        state.display_frusta_3d[frame_index],
        color="#e4572e",
        alpha=0.95,
        linewidth=1.8,
    )
    _draw_current_axes_3d(
        ax,
        state.poses[frame_index],
        camera_convention=state.target_convention,
        depth_m=state.frustum_depth_m,
    )
    ax.scatter(
        state.display_centers_3d[frame_index, 0],
        state.display_centers_3d[frame_index, 1],
        state.display_centers_3d[frame_index, 2],
        color="#e4572e",
        s=36,
        depthshade=False,
    )

    y_label = "Forward Z (m)" if state.target_convention == "OpenCV" else "Forward -Z (m)"
    z_label = "Up -Y (m)" if state.target_convention == "OpenCV" else "Up Y (m)"
    ax.set_xlabel("X right (m)")
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    frame = camera_info.frames[frame_index]
    current_center = state.centers[frame_index]
    current_forward = current_center[2] if state.target_convention == "OpenCV" else -current_center[2]
    current_up = -current_center[1] if state.target_convention == "OpenCV" else current_center[1]
    ax.set_title(
        f"3D Frustum View ({state.target_convention})\n"
        f"sample={frame.sample_index}  t={frame.timestamp_sec:.3f}s",
        fontsize=12,
    )
    ax.text2D(
        0.02,
        0.98,
        (
            f"x={current_center[0]:.3f} m\n"
            f"forward={current_forward:.3f} m\n"
            f"up={current_up:.3f} m\n"
            f"source_frame={frame.source_frame_index}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.35"},
    )


def render_visualization_frames(
    camera_info: CameraInfoDocument,
    state: VisualizationState,
    output_dir: Path,
    *,
    view_mode: str,
    max_history_frustums: int,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = camera_info.image_size
    dpi = 100
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")

    try:
        for frame_index, frame in enumerate(camera_info.frames):
            figure.clf()
            if view_mode == "2d":
                axis = figure.add_axes([0.08, 0.08, 0.88, 0.84])
                _render_2d_frame(
                    ax=axis,
                    frame_index=frame_index,
                    camera_info=camera_info,
                    state=state,
                    max_history_frustums=max_history_frustums,
                )
            else:
                axis = figure.add_axes([0.04, 0.06, 0.90, 0.86], projection="3d")
                _render_3d_frame(
                    ax=axis,
                    frame_index=frame_index,
                    camera_info=camera_info,
                    state=state,
                    max_history_frustums=max_history_frustums,
                )

            output_path = output_dir / f"frame_{frame.sample_index:06d}.png"
            figure.savefig(output_path, dpi=dpi, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)

    return tuple(sorted(output_dir.glob("frame_*.png")))


def create_pose_visualization(config: PoseVisualizationConfig) -> Path:
    ensure_ffmpeg_available()
    camera_info = load_camera_info(config.camera_json_path)
    target_output = (
        config.output_video_path.expanduser().resolve()
        if config.output_video_path is not None
        else build_default_visualization_output_path(config.camera_json_path, config.view_mode)
    )
    output_fps = config.output_fps or camera_info.sample_fps
    state = build_visualization_state(
        camera_info,
        visualization_convention=config.visualization_convention,
        frustum_scale=config.frustum_scale,
        history_window_sec=config.history_window_sec,
    )

    temp_dir = _build_temp_dir(f"{camera_info.video_id}-{config.view_mode}-viz-", config.temp_root_dir)
    right_frames_dir = temp_dir / "right_frames"
    left_frames_dir = temp_dir / "left_frames"
    LOGGER.info(
        "video_id=%s view_mode=%s output=%s temp_dir=%s",
        camera_info.video_id,
        config.view_mode,
        target_output,
        temp_dir,
    )

    try:
        sampled_video_frames = _extract_sampled_video_frames(camera_info, right_frames_dir)
        LOGGER.info("extracted_right_frames=%s", len(sampled_video_frames))
        rendered_frames = render_visualization_frames(
            camera_info,
            state,
            left_frames_dir,
            view_mode=config.view_mode,
            max_history_frustums=config.max_history_frustums,
        )
        LOGGER.info("rendered_left_frames=%s", len(rendered_frames))
        _compose_visualization_video(left_frames_dir, right_frames_dir, target_output, fps=output_fps)
        return target_output
    finally:
        _cleanup_temp_dir(temp_dir, config.keep_temp)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one *-camera.json file by rendering camera frusta on the left and "
            "the sampled source video on the right."
        )
    )
    parser.add_argument("--camera-json", required=True, help="Path to one *-camera.json file.")
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional output MP4 path. Defaults to <camera-json-stem>-<view-mode>-visualization.mp4.",
    )
    parser.add_argument(
        "--view-mode",
        choices=["2d", "3d"],
        default="3d",
        help="Visualization mode for the left panel.",
    )
    parser.add_argument(
        "--visualization-convention",
        choices=sorted(SUPPORTED_CAMERA_CONVENTIONS),
        default=None,
        help="Target convention used to draw the frusta. Defaults to the convention stored in the JSON.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Output video FPS. Defaults to the sample_fps stored in the JSON.",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the rendered frustum size.",
    )
    parser.add_argument(
        "--history-window-sec",
        type=float,
        default=10.0,
        help="Only draw recent trajectory/frusta within this time window.",
    )
    parser.add_argument(
        "--max-history-frustums",
        type=int,
        default=80,
        help="Maximum number of historical frusta to draw together with the current one.",
    )
    parser.add_argument(
        "--temp-root-dir",
        default=None,
        help="Optional temp directory root for sampled frames and rendered panels.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate left/right frame directories for inspection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    output_path = create_pose_visualization(
        PoseVisualizationConfig(
            camera_json_path=Path(args.camera_json),
            output_video_path=Path(args.output_video) if args.output_video else None,
            view_mode=args.view_mode,
            visualization_convention=args.visualization_convention,
            output_fps=args.output_fps,
            frustum_scale=args.frustum_scale,
            history_window_sec=args.history_window_sec,
            max_history_frustums=args.max_history_frustums,
            keep_temp=bool(args.keep_temp),
            temp_root_dir=Path(args.temp_root_dir) if args.temp_root_dir else None,
        )
    )
    print(json.dumps({"output_video": str(output_path)}, ensure_ascii=False))
    return 0


__all__ = [
    "CameraInfoDocument",
    "CameraInfoFrame",
    "PoseVisualizationConfig",
    "build_default_visualization_output_path",
    "build_visualization_state",
    "convert_camera_pose_convention",
    "create_pose_visualization",
    "load_camera_info",
    "main",
    "render_visualization_frames",
]
