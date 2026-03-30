from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video2poses.adaptive_scheduler import AdaptiveInferenceController
from video2poses.chunk_planner import build_chunk
from video2poses.image_pose_cli import CameraIntrinsics, CoordinateSystemSpec, ImagePoseResult, PoseEstimate
from video2poses.video_pipeline import process_video
from video2poses.video_pose_types import FrameRecord, VideoJob, VideoMetadata, VideoPoseConfig


def _matrix(tx: float, ty: float = 0.0) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, tx),
        (0.0, 1.0, 0.0, ty),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _pose(camera_pose) -> PoseEstimate:
    return PoseEstimate(
        image_path="/tmp/fake.png",
        cam_quats=(0.0, 0.0, 0.0, 1.0),
        cam_trans=(camera_pose[0][3], camera_pose[1][3], camera_pose[2][3]),
        camera_pose=camera_pose,
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
        views=tuple(_pose(pose) for pose in poses),
    )


class FakeClient:
    def infer(self, image_paths):
        raise AssertionError("process_video should use patched infer_next_chunk")

    def health(self):
        return None


class VideoPipelineTests(unittest.TestCase):
    @patch("video2poses.video_pipeline.ensure_ffmpeg_available")
    @patch("video2poses.video_pipeline.probe_video")
    @patch("video2poses.video_pipeline.extract_frames")
    @patch("video2poses.video_pipeline.infer_next_chunk")
    def test_process_video_writes_json(
        self,
        mocked_infer_next_chunk,
        mocked_extract_frames,
        mocked_probe_video,
        mocked_ensure_ffmpeg,
    ) -> None:
        del mocked_ensure_ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "demo.mp4"
            video_path.write_bytes(b"video")
            output_path = root / "demo-camera.json"

            frames = [
                FrameRecord(0, 0, 0.0, root / "frame0.png"),
                FrameRecord(1, 1, 0.5, root / "frame1.png"),
                FrameRecord(2, 2, 1.0, root / "frame2.png"),
            ]
            mocked_extract_frames.return_value = frames
            mocked_probe_video.return_value = VideoMetadata(
                source_video_fps=30.0,
                duration_sec=1.0,
                total_frames=3,
                width=640,
                height=480,
            )

            chunk0 = build_chunk("demo", frames, cursor=0, chunk_size=2, chunk_index=0)
            chunk1 = build_chunk("demo", frames, cursor=2, chunk_size=2, chunk_index=1)
            assert chunk0 is not None
            assert chunk1 is not None
            mocked_infer_next_chunk.side_effect = [
                (chunk0, _result(_matrix(0.0), _matrix(1.0))),
                (chunk1, _result(_matrix(10.0), _matrix(11.0))),
            ]

            config = VideoPoseConfig(
                input_dir=root,
                output_dir=root,
                server_url="http://127.0.0.1:18080",
                sample_fps=2.0,
                initial_max_inference_frames=2,
                min_inference_frames=2,
                initial_max_concurrent=1,
                max_video_workers=1,
                max_retries=2,
                request_timeout_sec=30.0,
            )
            controller = AdaptiveInferenceController(
                initial_chunk_size=2,
                min_chunk_size=2,
                initial_concurrency=1,
            )
            job = VideoJob(video_id="demo", video_path=video_path, output_path=output_path)

            result_path = process_video(
                job=job,
                config=config,
                client=FakeClient(),
                controller=controller,
            )

            self.assertEqual(result_path, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["video_id"], "demo")
            self.assertEqual(payload["num_sampled_frames"], 3)
            self.assertEqual([frame["sample_index"] for frame in payload["frames"]], [0, 1, 2])
            self.assertEqual(payload["frames"][2]["pose"]["cam_trans"], [2.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
