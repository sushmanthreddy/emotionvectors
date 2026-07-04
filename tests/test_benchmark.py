from __future__ import annotations

from types import SimpleNamespace

import torch

from emotion_vectors.benchmark import run_benchmarks


def test_cpu_benchmark_discloses_actual_fallback_backend() -> None:
    config = SimpleNamespace(
        seed=0,
        token_start=50,
        use_kernels=True,
        activation_percentile=90,
    )

    report = run_benchmarks(config, device=torch.device("cpu"), iterations=1)

    masked = report["masked_mean"]
    assert masked["allclose"] is True
    assert masked["kernel_requested"] is True
    assert masked["kernel_available"] is False
    assert masked["implementation"] == "torch"
    assert masked["kernel_error"] is None
    assert masked["speedup"] is None
