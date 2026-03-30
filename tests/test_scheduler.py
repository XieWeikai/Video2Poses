from __future__ import annotations

import unittest

from video2poses.adaptive_scheduler import AdaptiveInferenceController


class AdaptiveSchedulerTests(unittest.TestCase):
    def test_success_grows_chunk_and_concurrency(self) -> None:
        controller = AdaptiveInferenceController(
            initial_chunk_size=8,
            min_chunk_size=2,
            initial_concurrency=2,
            success_window=2,
        )
        controller.after_success(latency_sec=1.0, batch_size=8)
        self.assertEqual(controller.current_chunk_size(), 8)
        self.assertEqual(controller.current_max_concurrency(), 2)

        controller.after_success(latency_sec=1.0, batch_size=8)
        self.assertEqual(controller.current_chunk_size(), 10)
        self.assertEqual(controller.current_max_concurrency(), 3)

    def test_high_latency_success_does_not_grow(self) -> None:
        controller = AdaptiveInferenceController(
            initial_chunk_size=8,
            min_chunk_size=2,
            initial_concurrency=2,
            success_window=2,
            growth_max_latency_sec=5.0,
            growth_max_latency_per_frame_sec=0.5,
            degrade_latency_sec=20.0,
            degrade_latency_per_frame_sec=2.0,
        )
        controller.after_success(latency_sec=6.0, batch_size=8)
        controller.after_success(latency_sec=6.0, batch_size=8)
        self.assertEqual(controller.current_chunk_size(), 8)
        self.assertEqual(controller.current_max_concurrency(), 2)

    def test_severe_success_latency_triggers_backpressure(self) -> None:
        controller = AdaptiveInferenceController(
            initial_chunk_size=16,
            min_chunk_size=2,
            initial_concurrency=4,
            degrade_latency_sec=30.0,
            degrade_latency_per_frame_sec=1.0,
        )
        controller.after_success(latency_sec=80.0, batch_size=20)
        self.assertEqual(controller.current_chunk_size(), 8)
        self.assertEqual(controller.current_max_concurrency(), 2)

    def test_growth_cooldown_blocks_immediate_regrowth(self) -> None:
        controller = AdaptiveInferenceController(
            initial_chunk_size=16,
            min_chunk_size=2,
            initial_concurrency=4,
            success_window=2,
            degrade_latency_sec=30.0,
            degrade_latency_per_frame_sec=1.0,
        )
        controller.after_success(latency_sec=35.0, batch_size=32)
        self.assertEqual(controller.current_chunk_size(), 8)
        self.assertEqual(controller.current_max_concurrency(), 3)

        controller.after_success(latency_sec=1.0, batch_size=8)
        controller.after_success(latency_sec=1.0, batch_size=8)
        self.assertEqual(controller.current_chunk_size(), 8)
        self.assertEqual(controller.current_max_concurrency(), 3)

    def test_failure_halves_capacity(self) -> None:
        controller = AdaptiveInferenceController(
            initial_chunk_size=10,
            min_chunk_size=2,
            initial_concurrency=4,
        )
        decision = controller.after_failure(attempt=1, batch_size=10)
        self.assertTrue(decision.should_retry)
        self.assertGreaterEqual(decision.backoff_sec, 1.0)
        self.assertEqual(controller.current_chunk_size(), 5)
        self.assertEqual(controller.current_max_concurrency(), 2)


if __name__ == "__main__":
    unittest.main()
