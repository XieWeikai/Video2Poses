from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .video_pose_types import VideoJob, VideoMetadata


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def ensure_ffmpeg_available() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Required executables not found in PATH: {', '.join(missing)}")


def build_output_dir(input_dir: Path) -> Path:
    root = input_dir.expanduser().resolve()
    return root.parent / f"{root.name}-camera-info"


def build_output_path(output_dir: Path, video_path: Path) -> Path:
    return output_dir / f"{video_path.stem}-camera.json"


def discover_videos(input_dir: Path, output_dir: Path | None = None) -> list[VideoJob]:
    root = input_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    resolved_output_dir = (output_dir or build_output_dir(root)).expanduser().resolve()
    videos = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"No video files found in {root}")

    return [
        VideoJob(
            video_id=video_path.stem,
            video_path=video_path.resolve(),
            output_path=build_output_path(resolved_output_dir, video_path.resolve()),
        )
        for video_path in videos
    ]


def _parse_fps(text: str) -> float:
    value = (text or "").strip()
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    return float(Fraction(value))


def probe_video(video_path: Path) -> VideoMetadata:
    path = video_path.expanduser().resolve()
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "ffprobe failed"
        raise RuntimeError(f"Failed to probe video {path}: {stderr}")

    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"No video stream found in {path}")

    fps = _parse_fps(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0"))
    duration_text = (
        video_stream.get("duration")
        or payload.get("format", {}).get("duration")
        or "0"
    )
    duration_sec = float(duration_text)
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    nb_frames = video_stream.get("nb_frames")
    if nb_frames not in (None, "", "N/A"):
        total_frames = int(nb_frames)
    else:
        total_frames = int(round(duration_sec * fps)) if fps > 0 and duration_sec > 0 else 0

    return VideoMetadata(
        source_video_fps=fps,
        duration_sec=duration_sec,
        total_frames=total_frames,
        width=width,
        height=height,
    )
