#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from video2poses.batch_pipeline import process_video_dir
from video2poses.video_pose_types import VideoPoseConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer camera poses and intrinsics for videos.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--sample-fps", type=float, required=True)
    parser.add_argument("--initial-max-inference-frames", type=int, default=32)
    parser.add_argument("--min-inference-frames", type=int, default=2)
    parser.add_argument("--initial-max-concurrent", type=int, default=2)
    parser.add_argument("--max-video-workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--output-dir")
    parser.add_argument("--temp-root-dir")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--max-inference-frames-cap", type=int)
    parser.add_argument("--max-concurrency-cap", type=int)
    parser.add_argument("--success-window", type=int, default=3)
    parser.add_argument("--base-backoff-sec", type=float, default=1.0)
    parser.add_argument("--max-backoff-sec", type=float, default=30.0)
    parser.add_argument("--growth-max-latency-sec", type=float, default=15.0)
    parser.add_argument("--growth-max-latency-per-frame-sec", type=float, default=0.75)
    parser.add_argument("--degrade-latency-sec", type=float, default=45.0)
    parser.add_argument("--degrade-latency-per-frame-sec", type=float, default=1.5)
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_config(args: argparse.Namespace) -> VideoPoseConfig:
    return VideoPoseConfig(
        input_dir=Path(args.input_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        server_url=args.server_url,
        sample_fps=args.sample_fps,
        initial_max_inference_frames=args.initial_max_inference_frames,
        min_inference_frames=args.min_inference_frames,
        initial_max_concurrent=args.initial_max_concurrent,
        max_video_workers=args.max_video_workers,
        max_retries=args.max_retries,
        request_timeout_sec=args.request_timeout_sec,
        keep_temp=args.keep_temp,
        temp_root_dir=Path(args.temp_root_dir).expanduser().resolve() if args.temp_root_dir else None,
        max_inference_frames_cap=args.max_inference_frames_cap,
        max_concurrency_cap=args.max_concurrency_cap,
        success_window=args.success_window,
        base_backoff_sec=args.base_backoff_sec,
        max_backoff_sec=args.max_backoff_sec,
        growth_max_latency_sec=args.growth_max_latency_sec,
        growth_max_latency_per_frame_sec=args.growth_max_latency_per_frame_sec,
        degrade_latency_sec=args.degrade_latency_sec,
        degrade_latency_per_frame_sec=args.degrade_latency_per_frame_sec,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = build_config(args)
    summary = process_video_dir(config.input_dir, config)
    print(json.dumps(summary.to_json_dict(), indent=2, ensure_ascii=False))
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
