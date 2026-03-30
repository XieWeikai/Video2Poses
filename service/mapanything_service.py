from __future__ import annotations

import argparse
import asyncio
import gc
import math
import os
import sys
import threading
import traceback
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VIDEO2POSES_ROOT = Path(__file__).resolve().parents[1]
MAPANYTHING_ROOT = VIDEO2POSES_ROOT / "third_party" / "map-anything"
MAPANYTHING_TRAIN_CONFIG = MAPANYTHING_ROOT / "configs" / "train.yaml"


def _parse_device_name(device_name: str) -> str:
    lowered = device_name.strip().lower()
    if lowered.startswith("cuda:"):
        return lowered
    if lowered.startswith("cuda") and lowered[4:].isdigit():
        return f"cuda:{lowered[4:]}"
    raise ValueError(
        f"Unsupported device string '{device_name}'. Use values like 'cuda0' or 'cuda:1'."
    )


def load_service_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Service config must be a YAML mapping.")

    service_cfg = config.setdefault("service", {})
    service_cfg.setdefault("host", "127.0.0.1")
    service_cfg.setdefault("port", 18080)
    service_cfg.setdefault("log_level", "info")
    service_cfg.setdefault("request_timeout_sec", 1800)

    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("model_dir", "/models/map-anything")

    devices = model_cfg.get("devices", config.get("devices"))
    if not devices:
        raise ValueError("Config must define at least one CUDA device in model.devices or devices.")
    model_cfg["devices"] = [_parse_device_name(device) for device in devices]

    infer_cfg = model_cfg.setdefault("infer", {})
    infer_cfg.setdefault("memory_efficient_inference", True)
    infer_cfg.setdefault("minibatch_size", None)
    infer_cfg.setdefault("use_amp", True)
    infer_cfg.setdefault("amp_dtype", "bf16")
    infer_cfg.setdefault("apply_mask", True)
    infer_cfg.setdefault("mask_edges", True)
    infer_cfg.setdefault("apply_confidence_mask", False)
    infer_cfg.setdefault("confidence_percentile", 10)
    infer_cfg.setdefault("use_multiview_confidence", False)
    infer_cfg.setdefault("multiview_conf_depth_abs_thresh", 0.02)
    infer_cfg.setdefault("multiview_conf_depth_rel_thresh", 0.02)

    return config


class InferenceRequest(BaseModel):
    image_paths: list[str] = Field(..., min_length=1)


class ServiceInferenceError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _compute_resize_metadata(image_paths, find_closest_aspect_ratio, resolution_set=518):
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    original_sizes = []
    aspect_ratios = []
    for image_path in image_paths:
        img = exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = img.size
        original_sizes.append((width, height))
        aspect_ratios.append(width / height)

    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    processed_width, processed_height = find_closest_aspect_ratio(
        average_aspect_ratio, resolution_set
    )

    resize_metadata = []
    for image_path, (orig_w, orig_h) in zip(image_paths, original_sizes):
        scale = max(processed_width / orig_w, processed_height / orig_h) + 1e-8
        scaled_w = int(math.floor(orig_w * scale))
        scaled_h = int(math.floor(orig_h * scale))
        crop_left = (scaled_w - processed_width) // 2
        crop_top = (scaled_h - processed_height) // 2
        resize_metadata.append(
            {
                "image_path": image_path,
                "original_width": orig_w,
                "original_height": orig_h,
                "processed_width": processed_width,
                "processed_height": processed_height,
                "scale": float(scale),
                "crop_left": int(crop_left),
                "crop_top": int(crop_top),
            }
        )
    return resize_metadata


def _remap_intrinsics_to_original(intrinsics, meta):
    scale = meta["scale"]
    fx = intrinsics[0][0] / scale
    fy = intrinsics[1][1] / scale
    cx = (intrinsics[0][2] + meta["crop_left"]) / scale
    cy = (intrinsics[1][2] + meta["crop_top"]) / scale
    return fx, fy, cx, cy


def _worker_loop(
    worker_idx: int,
    device: str,
    config: dict[str, Any],
    conn,
) -> None:
    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.chdir(MAPANYTHING_ROOT)
        if str(MAPANYTHING_ROOT) not in sys.path:
            sys.path.insert(0, str(MAPANYTHING_ROOT))

        import torch

        from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local
        from mapanything.utils.image import find_closest_aspect_ratio, load_images

        if device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":", maxsplit=1)[1]))
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except Exception:
            pass

        local_config = {
            "path": str(MAPANYTHING_TRAIN_CONFIG),
            "checkpoint_path": str(Path(config["model"]["model_dir"]) / "model.safetensors"),
            "config_overrides": ["model=mapanything", "machine=aws"],
        }
        model = initialize_mapanything_local(local_config, device).eval()
        infer_cfg = dict(config["model"]["infer"])
        conn.send({"type": "ready", "worker_idx": worker_idx, "device": device})

        while True:
            message = conn.recv()
            if message is None or message.get("type") == "shutdown":
                break

            job_id = message["job_id"]
            image_paths = message["image_paths"]
            try:
                views = load_images(image_paths)
                for view in views:
                    for key, value in list(view.items()):
                        if torch.is_tensor(value):
                            view[key] = value.to(device)

                with torch.no_grad():
                    predictions = model.infer(views, **infer_cfg)

                resize_metadata = _compute_resize_metadata(
                    image_paths=image_paths,
                    find_closest_aspect_ratio=find_closest_aspect_ratio,
                    resolution_set=518,
                )

                output_views = []
                for image_path, prediction, meta in zip(image_paths, predictions, resize_metadata):
                    intrinsics = prediction["intrinsics"][0].detach().cpu().tolist()
                    fx, fy, cx, cy = _remap_intrinsics_to_original(intrinsics, meta)
                    camera_pose = prediction["camera_poses"][0].detach().cpu().tolist()
                    cam_quats = prediction["cam_quats"][0].detach().cpu().tolist()
                    cam_trans = prediction["cam_trans"][0].detach().cpu().tolist()
                    output_views.append(
                        {
                            "image_path": image_path,
                            "cam_quats": cam_quats,
                            "cam_trans": cam_trans,
                            "camera_pose": camera_pose,
                            "intrinsics": {
                                "fx": fx,
                                "fy": fy,
                                "cx": cx,
                                "cy": cy,
                            },
                            "original_image_size": [
                                meta["original_width"],
                                meta["original_height"],
                            ],
                            "processed_image_size": [
                                meta["processed_width"],
                                meta["processed_height"],
                            ],
                            "metric_scaling_factor": float(
                                prediction["metric_scaling_factor"][0].detach().cpu().item()
                            ),
                        }
                    )

                conn.send(
                    {
                        "type": "result",
                        "job_id": job_id,
                        "success": True,
                        "result": {
                            "device": device,
                            "coordinate_system": {
                                "camera_pose_type": "cam2world",
                                "camera_convention": "OpenCV",
                                "axes": "+X right, +Y down, +Z forward",
                            },
                            "views": output_views,
                        },
                    }
                )
            except Exception:
                error_text = traceback.format_exc()
                print(error_text, file=sys.stderr, flush=True)
                is_cuda_oom = (
                    "cuda out of memory" in error_text.lower()
                    or "cublas_status_alloc_failed" in error_text.lower()
                )
                if is_cuda_oom:
                    gc.collect()
                    try:
                        if device.startswith("cuda:"):
                            torch.cuda.set_device(int(device.split(":", maxsplit=1)[1]))
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    conn.send(
                        {
                            "type": "result",
                            "job_id": job_id,
                            "success": False,
                            "status_code": 503,
                            "code": "cuda_out_of_memory",
                            "message": (
                                f"CUDA out of memory on {device}. "
                                "The request failed, but the service stayed alive."
                            ),
                        }
                    )
                else:
                    conn.send(
                        {
                            "type": "result",
                            "job_id": job_id,
                            "success": False,
                            "status_code": 500,
                            "code": "inference_failed",
                            "message": error_text,
                        }
                    )
    except Exception:
        error_text = traceback.format_exc()
        print(error_text, file=sys.stderr, flush=True)
        try:
            conn.send(
                {
                    "type": "startup_error",
                    "worker_idx": worker_idx,
                    "device": device,
                    "message": error_text,
                }
            )
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class _WorkerState:
    process: Any
    conn: Any
    listener_thread: threading.Thread
    device: str
    pending: int = 0


class WorkerPool:
    def __init__(self, config: dict[str, Any]):
        import multiprocessing as mp

        self.config = config
        self._ctx = mp.get_context("spawn")
        self._workers: list[_WorkerState] = []
        self._pending_lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._futures_lock = threading.Lock()
        self._shutdown = threading.Event()

    def start(self) -> None:
        for worker_idx, device in enumerate(self.config["model"]["devices"]):
            parent_conn, child_conn = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_worker_loop,
                args=(worker_idx, device, self.config, child_conn),
                daemon=True,
            )
            process.start()
            child_conn.close()

            listener_thread = threading.Thread(
                target=self._listen_worker,
                args=(worker_idx, parent_conn),
                name=f"image-pose-listener-{worker_idx}",
                daemon=True,
            )
            state = _WorkerState(
                process=process,
                conn=parent_conn,
                listener_thread=listener_thread,
                device=device,
            )
            self._workers.append(state)

        startup_deadline = asyncio.get_running_loop().time() + 600
        ready_workers = set()
        while len(ready_workers) < len(self._workers):
            if asyncio.get_running_loop().time() > startup_deadline:
                raise RuntimeError("Timed out waiting for image-pose workers to start.")
            for idx, worker in enumerate(self._workers):
                if idx in ready_workers:
                    continue
                if worker.conn.poll(0.1):
                    message = worker.conn.recv()
                    if message["type"] == "ready":
                        ready_workers.add(idx)
                    elif message["type"] == "startup_error":
                        raise RuntimeError(message["message"])
        for worker in self._workers:
            worker.listener_thread.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        for worker in self._workers:
            try:
                worker.conn.send({"type": "shutdown"})
            except Exception:
                pass
        for worker in self._workers:
            worker.process.join(timeout=10)
            try:
                worker.conn.close()
            except Exception:
                pass

    def submit(self, image_paths: list[str]) -> Future:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        for image_path in image_paths:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image path not found: {image_path}")

        with self._pending_lock:
            worker_idx = min(
                range(len(self._workers)),
                key=lambda idx: self._workers[idx].pending,
            )
            self._workers[worker_idx].pending += 1

        future: Future = Future()
        job_id = uuid.uuid4().hex
        with self._futures_lock:
            self._futures[job_id] = future
        self._workers[worker_idx].conn.send(
            {"type": "job", "job_id": job_id, "image_paths": image_paths}
        )
        return future

    def _listen_worker(self, worker_idx: int, conn) -> None:
        while not self._shutdown.is_set():
            try:
                message = conn.recv()
            except EOFError:
                break
            except OSError:
                break

            if message["type"] != "result":
                continue

            with self._pending_lock:
                self._workers[worker_idx].pending = max(0, self._workers[worker_idx].pending - 1)

            with self._futures_lock:
                future = self._futures.pop(message["job_id"], None)
            if future is None:
                continue

            if message["success"]:
                future.set_result(message["result"])
            else:
                future.set_exception(
                    ServiceInferenceError(
                        status_code=int(message["status_code"]),
                        code=str(message["code"]),
                        message=str(message["message"]),
                    )
                )


def create_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Image Pose Service", version="0.1.0")
    pool = WorkerPool(config)
    timeout_sec = float(config["service"]["request_timeout_sec"])

    @app.on_event("startup")
    async def startup_event():
        pool.start()
        app.state.worker_pool = pool

    @app.on_event("shutdown")
    async def shutdown_event():
        pool.shutdown()

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "devices": [worker.device for worker in pool._workers],
            "pending": [worker.pending for worker in pool._workers],
        }

    @app.post("/infer")
    async def infer(request: InferenceRequest):
        normalized_paths = [str(Path(path).expanduser().resolve()) for path in request.image_paths]
        try:
            future = pool.submit(normalized_paths)
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_sec)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Inference request timed out.") from exc
        except ServiceInferenceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={
                    "code": exc.code,
                    "message": exc.message,
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "runtime_error",
                    "message": str(exc),
                },
            ) from exc

        return result

    return app


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the image-pose inference service.")
    parser.add_argument(
        "--config",
        default="configs/service/mapanything_infer_service.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--host", default=None, help="Override service host from config.")
    parser.add_argument("--port", type=int, default=None, help="Override service port from config.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override model.model_dir from config.",
    )
    return parser
