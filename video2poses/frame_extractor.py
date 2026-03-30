from __future__ import annotations

import subprocess
from pathlib import Path

from .video_pose_types import FrameRecord, VideoMetadata


def extract_frames(
    video_path: Path,
    temp_dir: Path,
    sample_fps: float,
    metadata: VideoMetadata,
) -> list[FrameRecord]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = temp_dir / "frame_%06d.png"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path.expanduser().resolve()),
            "-vf",
            f"fps={sample_fps}",
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
        raise RuntimeError(f"Failed to extract frames from {video_path}: {stderr}")

    image_paths = sorted(temp_dir.glob("frame_*.png"))
    if not image_paths:
        raise RuntimeError(f"No frames extracted from video {video_path}")

    frame_records: list[FrameRecord] = []
    for sample_index, image_path in enumerate(image_paths):
        timestamp_sec = sample_index / sample_fps
        source_frame_index = int(round(timestamp_sec * metadata.source_video_fps))
        frame_records.append(
            FrameRecord(
                sample_index=sample_index,
                source_frame_index=source_frame_index,
                timestamp_sec=timestamp_sec,
                image_path=image_path.resolve(),
            )
        )
    return frame_records
