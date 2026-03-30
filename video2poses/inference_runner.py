from __future__ import annotations

import logging
import time
import urllib.error

from .adaptive_scheduler import AdaptiveInferenceController
from .chunk_planner import build_chunk
from .image_pose_cli import ImagePoseClient, ImagePoseClientError, ImagePoseResult, ServiceHealth
from .video_pose_types import InferenceChunk


LOGGER = logging.getLogger(__name__)


def _safe_health(client: ImagePoseClient) -> ServiceHealth | None:
    try:
        return client.health()
    except Exception:
        return None


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, ImagePoseClientError):
        return exc.status_code in {502, 503, 504}
    return isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError))


def infer_next_chunk(
    *,
    client: ImagePoseClient,
    video_id: str,
    frames,
    cursor: int,
    chunk_index: int,
    controller: AdaptiveInferenceController,
    max_retries: int,
) -> tuple[InferenceChunk, ImagePoseResult]:
    attempt = 0
    while True:
        attempt += 1
        sleep_before_retry = 0.0
        should_retry = False
        chunk_size = controller.current_chunk_size()
        chunk = build_chunk(
            video_id=video_id,
            frames=frames,
            cursor=cursor,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
        )
        if chunk is None:
            raise ValueError("No chunk available for the current cursor")

        controller.acquire_slot()
        started = time.monotonic()
        try:
            result = client.infer(chunk.image_paths)
            latency_sec = time.monotonic() - started
            health = _safe_health(client)
            controller.after_success(latency_sec=latency_sec, batch_size=len(chunk.frame_records), health=health)
            state = controller.current_state()
            LOGGER.info(
                "video=%s chunk=%s attempt=%s batch=%s latency_sec=%.3f next_chunk_size=%s next_concurrency=%s",
                video_id,
                chunk_index,
                attempt,
                len(chunk.frame_records),
                latency_sec,
                state.effective_chunk_size,
                state.effective_concurrency,
            )
            return chunk, result
        except Exception as exc:
            retryable = is_retryable_error(exc)
            health = _safe_health(client)
            if retryable and attempt <= max_retries:
                decision = controller.after_failure(
                    attempt=attempt,
                    batch_size=len(chunk.frame_records),
                    health=health,
                )
                state = controller.current_state()
                LOGGER.warning(
                    "video=%s chunk=%s attempt=%s batch=%s error=%s retry_in=%.2f next_chunk_size=%s next_concurrency=%s",
                    video_id,
                    chunk_index,
                    attempt,
                    len(chunk.frame_records),
                    exc,
                    decision.backoff_sec,
                    state.effective_chunk_size,
                    state.effective_concurrency,
                )
                sleep_before_retry = decision.backoff_sec
                should_retry = True
            else:
                raise
        finally:
            controller.release_slot()
        if should_retry:
            if sleep_before_retry > 0:
                time.sleep(sleep_before_retry)
            continue
