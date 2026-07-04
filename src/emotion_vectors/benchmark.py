"""Correctness and timing harness for extraction/V2 reduction paths."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import load_config
from .kernels import (
    masked_mean,
    masked_mean_fallback,
    masked_mean_triton,
    project_threshold,
    project_threshold_fallback,
    triton_masked_mean_available,
)


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark kernels against torch fallbacks")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--profile-model",
        action="store_true",
        help="also load the configured model and time a short all-layer capture",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    config = load_config(args.config, overrides=_overrides(args.set))
    device = _device(args.device)
    report = run_benchmarks(config, device=device, iterations=args.iterations)
    if args.profile_model:
        report["model_capture"] = _profile_model(config)

    output = config.resolve_path(config.paths.outputs) / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    (output / "bench_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = ["# Kernel benchmark", "", f"Device: `{device}`", ""]
    for name, values in report.items():
        markdown.extend(
            [
                f"## {name}",
                "",
                *(f"- {key}: `{value}`" for key, value in values.items()),
                "",
            ]
        )
    (output / "bench_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_benchmarks(config: Any, *, device: torch.device, iterations: int) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(int(config.seed))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    hidden = torch.randn(8, 192, 5120, device=device, dtype=dtype, generator=generator)
    mask = torch.ones(8, 192, device=device, dtype=torch.bool)
    mask[0, :17] = False

    fallback_mean = masked_mean_fallback(hidden, mask, config.token_start)
    kernel_requested = bool(config.use_kernels)
    kernel_available = triton_masked_mean_available(hidden)
    masked_mean_implementation = "torch"
    kernel_error: str | None = None
    if kernel_requested and kernel_available:
        try:
            triton_mean = masked_mean_triton(hidden, mask, config.token_start)
            torch.testing.assert_close(triton_mean, fallback_mean, rtol=3e-3, atol=3e-3)
            masked_mean_implementation = "triton"
        except Exception as exc:  # Triton compilation errors are backend-specific.
            kernel_error = f"{type(exc).__name__}: {exc}"
            masked_mean_implementation = "torch_fallback_after_triton_error"
    dispatched_mean = masked_mean(hidden, mask, config.token_start, use_kernel=kernel_requested)
    torch.testing.assert_close(dispatched_mean, fallback_mean, rtol=3e-3, atol=3e-3)
    fallback_ms = _time(
        lambda: masked_mean_fallback(hidden, mask, config.token_start), iterations, device
    )
    dispatch_ms = _time(
        lambda: masked_mean(hidden, mask, config.token_start, use_kernel=kernel_requested),
        iterations,
        device,
    )

    activations = torch.randn(8, 192, 5120, device=device, dtype=dtype, generator=generator)
    vectors = torch.randn(30, 5120, device=device, dtype=dtype, generator=generator)
    fallback_projection = project_threshold_fallback(
        activations,
        vectors,
        config.activation_percentile,
        attention_mask=mask,
        return_thresholds=True,
    )
    dispatch_projection = project_threshold(
        activations,
        vectors,
        config.activation_percentile,
        attention_mask=mask,
        use_kernel=bool(config.use_kernels),
        return_thresholds=True,
    )
    for actual, expected in zip(dispatch_projection, fallback_projection, strict=True):
        torch.testing.assert_close(actual, expected)
    project_fallback_ms = _time(
        lambda: project_threshold_fallback(
            activations,
            vectors,
            config.activation_percentile,
            attention_mask=mask,
        ),
        iterations,
        device,
    )
    project_dispatch_ms = _time(
        lambda: project_threshold(
            activations,
            vectors,
            config.activation_percentile,
            attention_mask=mask,
            use_kernel=bool(config.use_kernels),
        ),
        iterations,
        device,
    )
    return {
        "masked_mean": {
            "fallback_ms": round(fallback_ms, 4),
            "dispatch_ms": round(dispatch_ms, 4),
            "speedup": (
                round(fallback_ms / dispatch_ms, 3)
                if masked_mean_implementation == "triton"
                else None
            ),
            "allclose": True,
            "kernel_requested": kernel_requested,
            "kernel_available": kernel_available,
            "implementation": masked_mean_implementation,
            "kernel_error": kernel_error,
        },
        "project_threshold": {
            "fallback_ms": round(project_fallback_ms, 4),
            "dispatch_ms": round(project_dispatch_ms, 4),
            "speedup": None,
            "allclose": True,
            "custom_kernel_benchmarked": False,
            "implementation": (
                "torch matmul + quantile; no custom percentile kernel because profiling "
                "did not identify this path as a worthwhile fusion target"
            ),
        },
    }


def _time(function: Callable[[], Any], iterations: int, device: torch.device) -> float:
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def _profile_model(config: Any) -> dict[str, Any]:
    from .model import load_model

    bundle = load_model(config)
    prompt = "Human: State one neutral fact about water.\n\nAssistant:"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
        capture = bundle.encode_and_capture([prompt], token_start=0, collect_norms=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    shape = tuple(getattr(capture, "vectors", capture).shape)
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    return {
        "milliseconds": round(elapsed, 3),
        "capture_shape": shape,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "top_hotspots": profile.key_averages().table(sort_by=sort_key, row_limit=10),
        "expected_hotspot": "transformer forward",
    }


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _overrides(items: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        result[key] = yaml.safe_load(value)
    return result
