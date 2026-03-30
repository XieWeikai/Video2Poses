from __future__ import annotations

import unittest

import numpy as np

from video2poses.image_pose_cli import CameraIntrinsics, CoordinateSystemSpec, ImagePoseResult, PoseEstimate
from video2poses.pose_alignment import align_first_chunk, align_next_chunk, matrix_to_pose_fields
from video2poses.video_pose_types import FrameRecord, InferenceChunk


def _matrix(tx: float, ty: float = 0.0, tz: float = 0.0) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, tx),
        (0.0, 1.0, 0.0, ty),
        (0.0, 0.0, 1.0, tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _pose_estimate(image_path: str, camera_pose) -> PoseEstimate:
    cam_quats, cam_trans, normalized_pose = matrix_to_pose_fields(camera_pose)
    return PoseEstimate(
        image_path=image_path,
        cam_quats=cam_quats,
        cam_trans=cam_trans,
        camera_pose=normalized_pose,
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=320.0, cy=240.0),
        original_image_size=(640, 480),
        processed_image_size=(320, 240),
        metric_scaling_factor=1.0,
    )


def _result(*poses) -> ImagePoseResult:
    return ImagePoseResult(
        device="cuda:0",
        coordinate_system=CoordinateSystemSpec(
            camera_pose_type="cam2world",
            camera_convention="OpenCV",
            axes="+X right, +Y down, +Z forward",
        ),
        views=tuple(_pose_estimate(f"/tmp/{index}.png", pose) for index, pose in enumerate(poses)),
    )


class PoseAlignmentTests(unittest.TestCase):
    def test_matrix_to_pose_fields_identity(self) -> None:
        quats, trans, pose = matrix_to_pose_fields(_matrix(0.0))
        self.assertEqual(quats, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(trans, (0.0, 0.0, 0.0))
        self.assertEqual(pose, _matrix(0.0))

    def test_matrix_to_pose_fields_accepts_numpy(self) -> None:
        quats, trans, pose = matrix_to_pose_fields(np.asarray(_matrix(2.0), dtype=np.float64))
        self.assertEqual(quats, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(trans, (2.0, 0.0, 0.0))
        self.assertEqual(pose, _matrix(2.0))

    def test_align_chunks_with_shared_anchor(self) -> None:
        chunk0 = InferenceChunk(
            video_id="demo",
            chunk_index=0,
            start_sample_index=0,
            frame_records=(
                FrameRecord(0, 0, 0.0, image_path=None),  # type: ignore[arg-type]
                FrameRecord(1, 1, 0.5, image_path=None),  # type: ignore[arg-type]
            ),
            has_overlap_anchor=False,
        )
        aligned0 = align_first_chunk(chunk0, _result(_matrix(0.0), _matrix(1.0)))
        self.assertEqual(len(aligned0.aligned_frames), 2)
        self.assertEqual(aligned0.anchor_global_pose, _matrix(1.0))

        chunk1 = InferenceChunk(
            video_id="demo",
            chunk_index=1,
            start_sample_index=2,
            frame_records=(
                FrameRecord(1, 1, 0.5, image_path=None),  # type: ignore[arg-type]
                FrameRecord(2, 2, 1.0, image_path=None),  # type: ignore[arg-type]
            ),
            has_overlap_anchor=True,
        )
        aligned1 = align_next_chunk(chunk1, _result(_matrix(10.0), _matrix(11.0, 2.0)), aligned0.anchor_global_pose)
        self.assertEqual(len(aligned1.aligned_frames), 1)
        frame = aligned1.aligned_frames[0]
        self.assertEqual(frame.sample_index, 2)
        self.assertAlmostEqual(frame.cam_trans[0], 2.0)
        self.assertAlmostEqual(frame.cam_trans[1], 2.0)
        self.assertAlmostEqual(frame.cam_trans[2], 0.0)
        self.assertEqual(aligned1.anchor_global_pose, frame.camera_pose)


if __name__ == "__main__":
    unittest.main()
