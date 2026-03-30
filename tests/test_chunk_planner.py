from __future__ import annotations

import unittest
from pathlib import Path

from video2poses.chunk_planner import advance_cursor, build_chunk, plan_chunks
from video2poses.video_pose_types import FrameRecord


def _make_frames(count: int) -> list[FrameRecord]:
    return [
        FrameRecord(
            sample_index=index,
            source_frame_index=index * 3,
            timestamp_sec=index * 0.5,
            image_path=Path(f"/tmp/frame_{index:06d}.png"),
        )
        for index in range(count)
    ]


class ChunkPlannerTests(unittest.TestCase):
    def test_plan_chunks_with_overlap(self) -> None:
        frames = _make_frames(11)
        chunks = plan_chunks(frames, chunk_size=5, video_id="demo")

        self.assertEqual(len(chunks), 3)
        self.assertEqual([frame.sample_index for frame in chunks[0].frame_records], [0, 1, 2, 3, 4])
        self.assertEqual([frame.sample_index for frame in chunks[1].frame_records], [4, 5, 6, 7, 8])
        self.assertEqual([frame.sample_index for frame in chunks[2].frame_records], [8, 9, 10])
        self.assertFalse(chunks[0].has_overlap_anchor)
        self.assertTrue(chunks[1].has_overlap_anchor)
        self.assertTrue(chunks[2].has_overlap_anchor)

    def test_build_chunk_and_advance_cursor(self) -> None:
        frames = _make_frames(6)
        chunk0 = build_chunk(video_id="demo", frames=frames, cursor=0, chunk_size=4, chunk_index=0)
        self.assertIsNotNone(chunk0)
        assert chunk0 is not None
        cursor = advance_cursor(0, chunk0)
        self.assertEqual(cursor, 4)

        chunk1 = build_chunk(video_id="demo", frames=frames, cursor=cursor, chunk_size=4, chunk_index=1)
        self.assertIsNotNone(chunk1)
        assert chunk1 is not None
        self.assertEqual([frame.sample_index for frame in chunk1.frame_records], [3, 4, 5])
        self.assertEqual(advance_cursor(cursor, chunk1), 6)


if __name__ == "__main__":
    unittest.main()
