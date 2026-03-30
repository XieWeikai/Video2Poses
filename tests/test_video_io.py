from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video2poses.video_io import build_output_dir, build_output_path, discover_videos, probe_video


class VideoIoTests(unittest.TestCase):
    def test_discover_videos_and_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.mp4").write_bytes(b"")
            (root / "b.mov").write_bytes(b"")
            (root / "note.txt").write_text("ignored", encoding="utf-8")

            output_dir = build_output_dir(root)
            jobs = discover_videos(root)
            self.assertEqual(output_dir.name, f"{root.name}-camera-info")
            self.assertEqual([job.video_id for job in jobs], ["a", "b"])
            self.assertEqual(build_output_path(output_dir, root / "a.mp4"), output_dir / "a-camera.json")

    @patch("video2poses.video_io.subprocess.run")
    def test_probe_video_parses_ffprobe_json(self, mocked_run) -> None:
        mocked_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=(
                '{"streams":[{"codec_type":"video","avg_frame_rate":"30000/1001",'
                '"width":1280,"height":720,"nb_frames":"120"}],'
                '"format":{"duration":"4.0"}}'
            ),
            stderr="",
        )
        metadata = probe_video(Path("/tmp/demo.mp4"))
        self.assertAlmostEqual(metadata.source_video_fps, 30000 / 1001, places=6)
        self.assertEqual(metadata.total_frames, 120)
        self.assertEqual(metadata.width, 1280)
        self.assertEqual(metadata.height, 720)
        self.assertAlmostEqual(metadata.duration_sec, 4.0)


if __name__ == "__main__":
    unittest.main()
