from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video2poses.pose_visualizer import (
    build_default_visualization_output_path,
    build_visualization_state,
    convert_camera_pose_convention,
    load_camera_info,
    render_visualization_frames,
)


def _camera_payload(source_video: str) -> dict:
    return {
        "schema_version": "v1",
        "video_id": "demo",
        "source_video": source_video,
        "source_video_fps": 30.0,
        "sample_fps": 2.0,
        "num_sampled_frames": 2,
        "coordinate_system": {
            "camera_pose_type": "cam2world",
            "camera_convention": "OpenCV",
            "axes": "+X right, +Y down, +Z forward",
        },
        "frames": [
            {
                "sample_index": 0,
                "source_frame_index": 0,
                "timestamp_sec": 0.0,
                "image_size": [320, 180],
                "intrinsics": {
                    "fx": 300.0,
                    "fy": 300.0,
                    "cx": 160.0,
                    "cy": 90.0,
                },
                "pose": {
                    "camera_pose": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "cam_quats": [0.0, 0.0, 0.0, 1.0],
                    "cam_trans": [0.0, 0.0, 0.0],
                },
            },
            {
                "sample_index": 1,
                "source_frame_index": 15,
                "timestamp_sec": 0.5,
                "image_size": [320, 180],
                "intrinsics": {
                    "fx": 300.0,
                    "fy": 300.0,
                    "cx": 160.0,
                    "cy": 90.0,
                },
                "pose": {
                    "camera_pose": [
                        [1.0, 0.0, 0.0, 0.2],
                        [0.0, 1.0, 0.0, 0.1],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "cam_quats": [0.0, 0.0, 0.0, 1.0],
                    "cam_trans": [0.2, 0.1, 0.4],
                },
            },
        ],
    }


class PoseVisualizerTests(unittest.TestCase):
    def test_load_camera_info_parses_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            camera_json = root / "demo-camera.json"
            source_video = root / "demo.mp4"
            source_video.write_bytes(b"video")
            camera_json.write_text(
                json.dumps(_camera_payload(str(source_video)), ensure_ascii=False),
                encoding="utf-8",
            )

            document = load_camera_info(camera_json)

            self.assertEqual(document.video_id, "demo")
            self.assertEqual(document.source_video, source_video.resolve())
            self.assertEqual(document.coordinate_system.camera_convention, "OpenCV")
            self.assertEqual(document.frames[1].cam_trans, (0.2, 0.1, 0.4))

    def test_convert_camera_pose_convention_flips_y_and_z(self) -> None:
        pose = np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        converted = convert_camera_pose_convention(
            pose,
            from_convention="OpenCV",
            to_convention="OpenGL",
        )

        expected = np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, -3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(converted, expected)

    def test_render_visualization_frames_creates_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            camera_json = root / "demo-camera.json"
            source_video = root / "demo.mp4"
            source_video.write_bytes(b"video")
            camera_json.write_text(
                json.dumps(_camera_payload(str(source_video)), ensure_ascii=False),
                encoding="utf-8",
            )
            document = load_camera_info(camera_json)
            state = build_visualization_state(
                document,
                visualization_convention="OpenCV",
                frustum_scale=1.0,
                history_window_sec=10.0,
            )

            output_dir = root / "left_frames"
            frames = render_visualization_frames(
                document,
                state,
                output_dir,
                view_mode="2d",
                max_history_frustums=8,
            )

            self.assertEqual(len(frames), 2)
            self.assertTrue(all(path.exists() for path in frames))

    def test_3d_display_projection_uses_forward_as_horizontal_and_up_as_vertical(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            camera_json = root / "demo-camera.json"
            source_video = root / "demo.mp4"
            source_video.write_bytes(b"video")
            camera_json.write_text(
                json.dumps(_camera_payload(str(source_video)), ensure_ascii=False),
                encoding="utf-8",
            )
            document = load_camera_info(camera_json)

            opencv_state = build_visualization_state(
                document,
                visualization_convention="OpenCV",
                frustum_scale=1.0,
                history_window_sec=10.0,
            )
            opengl_state = build_visualization_state(
                document,
                visualization_convention="OpenGL",
                frustum_scale=1.0,
                history_window_sec=10.0,
            )

            np.testing.assert_allclose(opencv_state.display_centers_3d[1], np.asarray([0.2, 0.4, -0.1]))
            np.testing.assert_allclose(opengl_state.display_centers_3d[1], np.asarray([0.2, 0.4, -0.1]))

    def test_history_window_limits_bounds_to_recent_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            camera_json = root / "demo-camera.json"
            source_video = root / "demo.mp4"
            source_video.write_bytes(b"video")
            payload = _camera_payload(str(source_video))
            payload["sample_fps"] = 1.0
            payload["num_sampled_frames"] = 3
            payload["frames"].append(
                {
                    "sample_index": 2,
                    "source_frame_index": 30,
                    "timestamp_sec": 2.0,
                    "image_size": [320, 180],
                    "intrinsics": {
                        "fx": 300.0,
                        "fy": 300.0,
                        "cx": 160.0,
                        "cy": 90.0,
                    },
                    "pose": {
                        "camera_pose": [
                            [1.0, 0.0, 0.0, 5.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "cam_quats": [0.0, 0.0, 0.0, 1.0],
                        "cam_trans": [5.0, 0.0, 0.0],
                    },
                }
            )
            camera_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            document = load_camera_info(camera_json)

            state = build_visualization_state(
                document,
                visualization_convention="OpenCV",
                frustum_scale=1.0,
                history_window_sec=1.0,
            )

            self.assertEqual(state.history_start_indices, (0, 1, 2))
            x_min, x_max, _, _ = state.window_bounds_2d[2]
            self.assertGreater(x_min, 4.0)
            self.assertGreater(x_max, 5.0)

    def test_default_output_path_uses_scripts_naming(self) -> None:
        camera_json = Path("/tmp/demo-camera.json")
        output_path = build_default_visualization_output_path(camera_json, "3d")
        self.assertEqual(output_path, Path("/tmp/demo-3d-visualization.mp4"))


if __name__ == "__main__":
    unittest.main()
