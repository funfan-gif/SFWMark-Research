"""Lightweight, algorithm-neutral runtime profiling for research stages."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


class RuntimeProfiler:
    """Measure wall time, CUDA peaks, and actual U-Net module invocations.

    ``torch_module`` is injectable so the CPU/unavailable-CUDA behavior can be
    checked without importing the full research environment.
    """

    def __init__(self, device="cuda", torch_module=None):
        if torch_module is None:
            import torch as torch_module

        self.torch = torch_module
        self.device = str(device)
        self.model_load_seconds = None
        self.processing_seconds = None
        self.inversion_processing_seconds = 0.0
        self.peak_cuda_allocated_gb = None
        self.peak_cuda_reserved_gb = None
        self.unet_forward_calls = 0
        self.num_images = 0
        self.requested_images = 0
        self.cache_hit_images = 0
        self.inversion_steps = 0
        self.total_refinement_iterations = 0
        self.total_newton_iterations = 0
        self._model_load_started = None
        self._processing_started = None
        self._hook_handle = None

    @property
    def cuda_available(self):
        cuda = getattr(self.torch, "cuda", None)
        return bool(
            self.device.startswith("cuda")
            and cuda is not None
            and cuda.is_available()
        )

    def synchronize(self):
        if self.cuda_available:
            self.torch.cuda.synchronize(self.device)

    def start_model_load(self):
        self.synchronize()
        self._model_load_started = time.perf_counter()

    def finish_model_load(self):
        if self._model_load_started is None:
            raise RuntimeError("start_model_load() must be called first")
        self.synchronize()
        self.model_load_seconds = time.perf_counter() - self._model_load_started
        return self.model_load_seconds

    def attach_unet(self, unet):
        if self._hook_handle is not None:
            raise RuntimeError("A U-Net counter is already attached")

        def count_forward(_module, _inputs):
            self.unet_forward_calls += 1

        self._hook_handle = unet.register_forward_pre_hook(count_forward)

    def start_processing(self):
        self.unet_forward_calls = 0
        if self.cuda_available:
            self.torch.cuda.reset_peak_memory_stats(self.device)
        self.synchronize()
        self._processing_started = time.perf_counter()

    def finish_processing(self):
        if self._processing_started is None:
            raise RuntimeError("start_processing() must be called first")
        self.synchronize()
        self.processing_seconds = time.perf_counter() - self._processing_started
        if self.cuda_available:
            scale = float(1024 ** 3)
            self.peak_cuda_allocated_gb = (
                self.torch.cuda.max_memory_allocated(self.device) / scale
            )
            self.peak_cuda_reserved_gb = (
                self.torch.cuda.max_memory_reserved(self.device) / scale
            )
        return self.processing_seconds

    def record_requested_images(self, count):
        self.requested_images += int(count)

    def record_cache_hits(self, count):
        self.cache_hit_images += int(count)

    def record_processed_images(self, count):
        self.num_images += int(count)

    def start_inversion(self):
        self.synchronize()
        return time.perf_counter()

    def finish_inversion(self, started_at):
        self.synchronize()
        self.inversion_processing_seconds += time.perf_counter() - started_at

    def record_inversion(self, method, num_images, num_steps, unet_calls_before):
        """Record one completed inversion call without changing its execution."""

        calls = self.unet_forward_calls - int(unet_calls_before)
        self.record_processed_images(num_images)
        self.inversion_steps += int(num_steps)
        if method == "exact_ddim":
            self.total_refinement_iterations += max(calls - int(num_steps), 0)
        elif method == "gnri":
            self.total_newton_iterations += calls

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def metrics(self):
        seconds_per_image = (
            self.processing_seconds / self.num_images
            if self.processing_seconds is not None and self.num_images > 0
            else None
        )
        calls_per_image = (
            self.unet_forward_calls / self.num_images
            if self.num_images > 0 else None
        )
        inversion_seconds_per_image = (
            self.inversion_processing_seconds / self.num_images
            if self.num_images > 0 and self.inversion_processing_seconds > 0 else None
        )
        return {
            "num_images": self.num_images,
            "requested_images": self.requested_images,
            "cache_hit_images": self.cache_hit_images,
            "model_load_seconds": self.model_load_seconds,
            "processing_seconds": self.processing_seconds,
            "seconds_per_image": seconds_per_image,
            "inversion_processing_seconds": self.inversion_processing_seconds,
            "inversion_seconds_per_image": inversion_seconds_per_image,
            "peak_cuda_allocated_gb": self.peak_cuda_allocated_gb,
            "peak_cuda_reserved_gb": self.peak_cuda_reserved_gb,
            "unet_forward_calls": self.unet_forward_calls,
            "unet_forward_calls_per_image": calls_per_image,
            "inversion_steps": self.inversion_steps,
            "total_refinement_iterations": self.total_refinement_iterations,
            "total_newton_iterations": self.total_newton_iterations,
        }

    def save(self, output_dir, metadata):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc).isoformat()
        payload = {
            **metadata,
            **self.metrics(),
            "runtime_recorded_at_utc": started_at,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        stage = str(metadata.get("stage", "unknown")).replace("/", "-")
        path = output_dir / f"runtime-{stage}-{digest}-{time.time_ns()}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved runtime profile: {path}")
        return path
