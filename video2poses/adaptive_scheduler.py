from __future__ import annotations

import random
import threading
from dataclasses import dataclass

from .image_pose_cli import ServiceHealth


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    backoff_sec: float


@dataclass(frozen=True)
class ControllerState:
    effective_chunk_size: int
    effective_concurrency: int


class AdaptiveInferenceController:
    def __init__(
        self,
        *,
        initial_chunk_size: int,
        min_chunk_size: int,
        initial_concurrency: int,
        max_chunk_size_cap: int | None = None,
        max_concurrency_cap: int | None = None,
        success_window: int = 3,
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 30.0,
        growth_max_latency_sec: float = 15.0,
        growth_max_latency_per_frame_sec: float = 0.75,
        degrade_latency_sec: float = 45.0,
        degrade_latency_per_frame_sec: float = 1.5,
    ) -> None:
        self._min_chunk_size = max(1, min_chunk_size)
        self._effective_chunk_size = max(self._min_chunk_size, initial_chunk_size)
        self._effective_concurrency = max(1, initial_concurrency)
        self._max_chunk_size_cap = max_chunk_size_cap or max(
            self._effective_chunk_size, self._effective_chunk_size * 2
        )
        self._max_concurrency_cap = max_concurrency_cap or max(
            self._effective_concurrency, self._effective_concurrency * 2
        )
        self._success_window = max(1, success_window)
        self._base_backoff_sec = max(0.0, base_backoff_sec)
        self._max_backoff_sec = max(self._base_backoff_sec, max_backoff_sec)
        self._growth_max_latency_sec = max(0.001, growth_max_latency_sec)
        self._growth_max_latency_per_frame_sec = max(0.001, growth_max_latency_per_frame_sec)
        self._degrade_latency_sec = max(self._growth_max_latency_sec, degrade_latency_sec)
        self._degrade_latency_per_frame_sec = max(
            self._growth_max_latency_per_frame_sec, degrade_latency_per_frame_sec
        )
        self._consecutive_successes = 0
        self._growth_cooldown_remaining = 0
        self._inflight = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def acquire_slot(self) -> None:
        with self._condition:
            while self._inflight >= self._effective_concurrency:
                self._condition.wait()
            self._inflight += 1

    def release_slot(self) -> None:
        with self._condition:
            self._inflight = max(0, self._inflight - 1)
            self._condition.notify_all()

    def current_chunk_size(self) -> int:
        with self._lock:
            return self._effective_chunk_size

    def current_max_concurrency(self) -> int:
        with self._lock:
            return self._effective_concurrency

    def current_state(self) -> ControllerState:
        with self._lock:
            return ControllerState(
                effective_chunk_size=self._effective_chunk_size,
                effective_concurrency=self._effective_concurrency,
            )

    def _health_has_low_pending(self, health: ServiceHealth | None) -> bool:
        if health is None:
            return True
        return sum(health.pending) <= self._effective_concurrency

    @staticmethod
    def _latency_per_frame(latency_sec: float, batch_size: int) -> float:
        return latency_sec / max(1, batch_size)

    def _should_reduce_for_latency(self, latency_sec: float, batch_size: int) -> bool:
        latency_per_frame = self._latency_per_frame(latency_sec, batch_size)
        return (
            latency_sec >= self._degrade_latency_sec
            or latency_per_frame >= self._degrade_latency_per_frame_sec
        )

    def _can_grow_for_latency(self, latency_sec: float, batch_size: int) -> bool:
        latency_per_frame = self._latency_per_frame(latency_sec, batch_size)
        return (
            latency_sec <= self._growth_max_latency_sec
            and latency_per_frame <= self._growth_max_latency_per_frame_sec
        )

    def _apply_failure_backpressure(self) -> None:
        self._effective_chunk_size = max(self._min_chunk_size, self._effective_chunk_size // 2)
        self._effective_concurrency = max(1, self._effective_concurrency // 2)
        self._condition.notify_all()

    def _enter_growth_cooldown(self, steps: int) -> None:
        self._growth_cooldown_remaining = max(self._growth_cooldown_remaining, max(0, steps))

    def _apply_success_latency_backpressure(self, *, latency_sec: float, batch_size: int) -> None:
        latency_per_frame = self._latency_per_frame(latency_sec, batch_size)
        severe = (
            latency_sec >= self._degrade_latency_sec * 2.0
            or latency_per_frame >= self._degrade_latency_per_frame_sec * 2.0
        )
        self._enter_growth_cooldown(self._success_window * 2)
        if severe:
            self._apply_failure_backpressure()
            return

        self._effective_chunk_size = max(self._min_chunk_size, self._effective_chunk_size // 2)
        if self._effective_concurrency > 1:
            self._effective_concurrency -= 1
        self._condition.notify_all()

    def after_success(self, *, latency_sec: float, batch_size: int, health: ServiceHealth | None = None) -> None:
        with self._lock:
            if self._should_reduce_for_latency(latency_sec, batch_size):
                self._consecutive_successes = 0
                self._apply_success_latency_backpressure(
                    latency_sec=latency_sec,
                    batch_size=batch_size,
                )
                return
            if self._growth_cooldown_remaining > 0:
                self._growth_cooldown_remaining -= 1
                self._consecutive_successes = 0
                return
            if not self._can_grow_for_latency(latency_sec, batch_size):
                self._consecutive_successes = 0
                return
            self._consecutive_successes += 1
            if self._consecutive_successes < self._success_window:
                return
            self._consecutive_successes = 0
            if batch_size >= self._effective_chunk_size and self._effective_chunk_size < self._max_chunk_size_cap:
                if self._health_has_low_pending(health):
                    step = max(1, min(4, self._effective_chunk_size // 4))
                    self._effective_chunk_size = min(self._max_chunk_size_cap, self._effective_chunk_size + step)
            if self._effective_concurrency < self._max_concurrency_cap and self._health_has_low_pending(health):
                self._effective_concurrency += 1
                self._condition.notify_all()

    def after_failure(
        self,
        *,
        attempt: int,
        batch_size: int,
        health: ServiceHealth | None = None,
    ) -> RetryDecision:
        del batch_size
        del health
        with self._lock:
            self._consecutive_successes = 0
            self._enter_growth_cooldown(self._success_window)
            self._apply_failure_backpressure()

        backoff = min(self._max_backoff_sec, self._base_backoff_sec * (2 ** max(0, attempt - 1)))
        if backoff > 0:
            backoff += random.uniform(0.0, min(1.0, backoff * 0.1))
        return RetryDecision(should_retry=True, backoff_sec=backoff)
