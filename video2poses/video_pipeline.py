from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from .adaptive_scheduler import AdaptiveInferenceController
from .chunk_planner import advance_cursor
from .frame_extractor import extract_frames
from .image_pose_cli import CoordinateSystemSpec, ImagePoseClient
from .inference_runner import infer_next_chunk
from .pose_alignment import align_first_chunk, align_next_chunk
from .video_io import ensure_ffmpeg_available, probe_video
from .video_pose_types import VideoCameraInfo, VideoJob, VideoPoseConfig


LOGGER = logging.getLogger(__name__)


def _write_video_camera_info(output_path: Path, camera_info: VideoCameraInfo) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    text = json.dumps(camera_info.to_json_dict(), indent=2, ensure_ascii=False) + "\n"
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(output_path)


def _build_temp_dir(job: VideoJob, config: VideoPoseConfig) -> Path:
    temp_root = config.temp_root_dir.expanduser().resolve() if config.temp_root_dir else None
    prefix = f"{job.video_id}-frames-"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(temp_root) if temp_root else None))


def _cleanup_temp_dir(temp_dir: Path, keep_temp: bool) -> None:
    if keep_temp:
        return
    shutil.rmtree(temp_dir, ignore_errors=True)


def process_video(
    *,
    job: VideoJob,
    config: VideoPoseConfig,
    client: ImagePoseClient,
    controller: AdaptiveInferenceController,
) -> Path:
    ensure_ffmpeg_available()
    metadata = probe_video(job.video_path)
    temp_dir = _build_temp_dir(job, config)
    LOGGER.info("video=%s temp_dir=%s", job.video_id, temp_dir)

    try:
        frames = extract_frames(job.video_path, temp_dir, config.sample_fps, metadata)
        cursor = 0
        chunk_index = 0
        aligned_frames = []
        coordinate_system: CoordinateSystemSpec | None = None
        previous_anchor_global_pose = None

        while cursor < len(frames):
            chunk, raw_result = infer_next_chunk(
                client=client,
                video_id=job.video_id,
                frames=frames,
                cursor=cursor,
                chunk_index=chunk_index,
                controller=controller,
                max_retries=config.max_retries,
            )
            if coordinate_system is None:
                coordinate_system = raw_result.coordinate_system
            elif raw_result.coordinate_system != coordinate_system:
                raise RuntimeError(
                    f"Coordinate system changed within video {job.video_id}: "
                    f"{coordinate_system} -> {raw_result.coordinate_system}"
                )

            if chunk_index == 0:
                aligned_chunk = align_first_chunk(chunk, raw_result)
            else:
                if previous_anchor_global_pose is None:
                    raise RuntimeError("Missing anchor pose for non-first chunk")
                aligned_chunk = align_next_chunk(chunk, raw_result, previous_anchor_global_pose)

            aligned_frames.extend(aligned_chunk.aligned_frames)
            previous_anchor_global_pose = aligned_chunk.anchor_global_pose
            cursor = advance_cursor(cursor, chunk)
            chunk_index += 1

        if coordinate_system is None:
            raise RuntimeError(f"No inference results produced for video {job.video_id}")

        camera_info = VideoCameraInfo(
            schema_version="v1",
            video_id=job.video_id,
            source_video=str(job.video_path),
            source_video_fps=metadata.source_video_fps,
            sample_fps=config.sample_fps,
            num_sampled_frames=len(aligned_frames),
            coordinate_system=coordinate_system,
            frames=tuple(aligned_frames),
        )
        _write_video_camera_info(job.output_path, camera_info)
        LOGGER.info("video=%s output=%s frames=%s", job.video_id, job.output_path, len(aligned_frames))
        return job.output_path
    finally:
        _cleanup_temp_dir(temp_dir, config.keep_temp)
