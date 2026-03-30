from __future__ import annotations

from typing import Iterable

import numpy as np

from .image_pose_cli import ImagePoseResult
from .video_pose_types import ChunkInferenceResult, FramePoseRecord, InferenceChunk, Matrix4


def _to_numpy_matrix4(camera_pose: Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(tuple(tuple(float(value) for value in row) for row in camera_pose), dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("camera_pose must be a 4x4 matrix")
    return matrix


def _to_matrix4_tuple(matrix: np.ndarray) -> Matrix4:
    if matrix.shape != (4, 4):
        raise ValueError("camera_pose must be a 4x4 matrix")
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())  # type: ignore[return-value]


def _matmul4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def _invert_rigid(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera_pose is not invertible") from exc


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")

    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = float(np.trace(rotation))

    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0)) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = float(np.sqrt(1.0 + m00 - m11 - m22)) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = float(np.sqrt(1.0 + m11 - m00 - m22)) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = float(np.sqrt(1.0 + m22 - m00 - m11)) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return (qx, qy, qz, qw)


def matrix_to_pose_fields(camera_pose: Matrix4) -> tuple[tuple[float, float, float, float], tuple[float, float, float], Matrix4]:
    matrix = _to_numpy_matrix4(camera_pose)
    rotation = matrix[:3, :3]
    translation = tuple(float(value) for value in matrix[:3, 3])
    quaternion = _rotation_matrix_to_quaternion_xyzw(rotation)
    return quaternion, translation, _to_matrix4_tuple(matrix)


def _align_frames(
    chunk: InferenceChunk,
    raw_result: ImagePoseResult,
    transform: np.ndarray,
    *,
    drop_anchor: bool,
) -> ChunkInferenceResult:
    if len(raw_result.views) != len(chunk.frame_records):
        raise ValueError("Service response count does not match chunk frame count")

    aligned_all: list[FramePoseRecord] = []
    for frame_record, view in zip(chunk.frame_records, raw_result.views):
        local_pose = _to_numpy_matrix4(view.camera_pose)
        global_pose = _matmul4(transform, local_pose)
        cam_quats, cam_trans, normalized_pose = matrix_to_pose_fields(global_pose)
        aligned_all.append(
            FramePoseRecord(
                sample_index=frame_record.sample_index,
                source_frame_index=frame_record.source_frame_index,
                timestamp_sec=frame_record.timestamp_sec,
                intrinsics=view.intrinsics,
                original_image_size=view.original_image_size,
                processed_image_size=view.processed_image_size,
                metric_scaling_factor=view.metric_scaling_factor,
                camera_pose=normalized_pose,
                cam_quats=cam_quats,
                cam_trans=cam_trans,
            )
        )

    if not aligned_all:
        raise ValueError("Aligned result is empty")

    anchor_global_pose = aligned_all[-1].camera_pose
    aligned_frames = tuple(aligned_all[1:] if drop_anchor else aligned_all)
    return ChunkInferenceResult(
        chunk=chunk,
        aligned_frames=aligned_frames,
        anchor_global_pose=anchor_global_pose,
    )


def align_first_chunk(chunk: InferenceChunk, raw_result: ImagePoseResult) -> ChunkInferenceResult:
    identity = np.eye(4, dtype=np.float64)
    return _align_frames(chunk, raw_result, identity, drop_anchor=False)


def align_next_chunk(
    chunk: InferenceChunk,
    raw_result: ImagePoseResult,
    previous_anchor_global_pose: Matrix4,
) -> ChunkInferenceResult:
    if not chunk.has_overlap_anchor:
        raise ValueError("Expected overlap anchor for non-first chunk alignment")
    if not raw_result.views:
        raise ValueError("Empty service response for non-first chunk")

    previous_anchor_global_pose_np = _to_numpy_matrix4(previous_anchor_global_pose)
    local_anchor = _to_numpy_matrix4(raw_result.views[0].camera_pose)
    transform = _matmul4(previous_anchor_global_pose_np, _invert_rigid(local_anchor))
    return _align_frames(chunk, raw_result, transform, drop_anchor=True)
