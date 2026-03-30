from __future__ import annotations

from .video_pose_types import FrameRecord, InferenceChunk


def build_chunk(
    video_id: str,
    frames: list[FrameRecord],
    cursor: int,
    chunk_size: int,
    chunk_index: int,
) -> InferenceChunk | None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if cursor < 0:
        raise ValueError("cursor must be >= 0")
    if cursor >= len(frames):
        return None

    has_overlap_anchor = cursor > 0 and chunk_size > 1
    start_index = cursor - 1 if has_overlap_anchor else cursor
    stop_index = min(len(frames), start_index + chunk_size)
    frame_records = tuple(frames[start_index:stop_index])
    if not frame_records:
        return None

    return InferenceChunk(
        video_id=video_id,
        chunk_index=chunk_index,
        start_sample_index=frames[cursor].sample_index,
        frame_records=frame_records,
        has_overlap_anchor=has_overlap_anchor,
    )


def advance_cursor(cursor: int, chunk: InferenceChunk) -> int:
    unique_count = len(chunk.frame_records) - (1 if chunk.has_overlap_anchor else 0)
    return cursor + unique_count


def plan_chunks(frames: list[FrameRecord], chunk_size: int, video_id: str = "video") -> list[InferenceChunk]:
    chunks: list[InferenceChunk] = []
    cursor = 0
    chunk_index = 0
    while True:
        chunk = build_chunk(video_id=video_id, frames=frames, cursor=cursor, chunk_size=chunk_size, chunk_index=chunk_index)
        if chunk is None:
            break
        chunks.append(chunk)
        cursor = advance_cursor(cursor, chunk)
        chunk_index += 1
    return chunks
