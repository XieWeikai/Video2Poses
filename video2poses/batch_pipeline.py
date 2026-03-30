from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .adaptive_scheduler import AdaptiveInferenceController
from .image_pose_cli import ImagePoseClient
from .video_io import build_output_dir, discover_videos, ensure_ffmpeg_available
from .video_pipeline import process_video
from .video_pose_types import BatchFailure, BatchSummary, VideoPoseConfig


LOGGER = logging.getLogger(__name__)


def process_video_dir(input_dir: Path, config: VideoPoseConfig) -> BatchSummary:
    ensure_ffmpeg_available()
    output_dir = (config.output_dir or build_output_dir(input_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_videos(input_dir, output_dir=output_dir)
    controller = AdaptiveInferenceController(
        initial_chunk_size=config.initial_max_inference_frames,
        min_chunk_size=config.min_inference_frames,
        initial_concurrency=config.initial_max_concurrent,
        max_chunk_size_cap=config.max_inference_frames_cap,
        max_concurrency_cap=config.max_concurrency_cap,
        success_window=config.success_window,
        base_backoff_sec=config.base_backoff_sec,
        max_backoff_sec=config.max_backoff_sec,
        growth_max_latency_sec=config.growth_max_latency_sec,
        growth_max_latency_per_frame_sec=config.growth_max_latency_per_frame_sec,
        degrade_latency_sec=config.degrade_latency_sec,
        degrade_latency_per_frame_sec=config.degrade_latency_per_frame_sec,
    )
    client = ImagePoseClient(
        server_url=config.server_url,
        request_timeout_sec=config.request_timeout_sec,
    )

    output_files: list[str] = []
    failures: list[BatchFailure] = []

    with ThreadPoolExecutor(max_workers=config.max_video_workers) as executor:
        future_to_job = {
            executor.submit(
                process_video,
                job=job,
                config=config,
                client=client,
                controller=controller,
            ): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                output_path = future.result()
                output_files.append(str(output_path))
            except Exception as exc:
                LOGGER.exception("video=%s failed", job.video_id)
                failures.append(
                    BatchFailure(
                        video_id=job.video_id,
                        video_path=str(job.video_path),
                        error=str(exc),
                    )
                )

    output_files.sort()
    failures.sort(key=lambda item: item.video_id)
    return BatchSummary(
        input_dir=str(input_dir.expanduser().resolve()),
        output_dir=str(output_dir),
        total_videos=len(jobs),
        succeeded=len(output_files),
        failed=len(failures),
        output_files=tuple(output_files),
        failures=tuple(failures),
    )
