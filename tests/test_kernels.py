from __future__ import annotations

import pytest
import torch

from emotion_vectors.kernels import (
    eligible_token_counts,
    masked_mean,
    masked_mean_fallback,
    project_threshold,
    project_threshold_fallback,
)


def test_masked_mean_matches_token_slice_and_padding() -> None:
    states = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],  # right padding
            [0, 1, 1, 1, 1, 1],  # left padding
        ]
    )

    actual = masked_mean_fallback(states, mask, token_start=2)
    expected = torch.stack([states[0, 2:4].mean(dim=0), states[1, 3:6].mean(dim=0)])

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(eligible_token_counts(mask, 2), torch.tensor([2, 3]))


def test_masked_mean_supports_per_row_starts_and_empty_rows() -> None:
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(3, 5, 4, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
    starts = torch.tensor([1, 2, 0])

    expected = masked_mean_fallback(states, mask, starts)
    dispatched = masked_mean(states, mask, starts, use_kernel=True)

    torch.testing.assert_close(dispatched, expected)
    torch.testing.assert_close(expected[0], states[0, 1:].mean(dim=0))
    torch.testing.assert_close(expected[1], states[1, 2:3].mean(dim=0))
    torch.testing.assert_close(expected[2], torch.zeros(4))


def test_masked_mean_accumulates_low_precision_in_float32() -> None:
    states = torch.tensor([[[1.0], [2.0], [4.0]]], dtype=torch.bfloat16)
    output = masked_mean_fallback(states, token_start=1)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.tensor([[3.0]]))


def test_project_threshold_matches_manual_projection_and_quantile() -> None:
    activations = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [100.0, 100.0]],
            [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]],
        ]
    )
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    projections, above, thresholds = project_threshold_fallback(
        activations,
        vectors,
        percentile=50,
        attention_mask=attention_mask,
        return_thresholds=True,
    )
    manual = activations.float() @ vectors.float().T
    valid = attention_mask.bool()
    manual_thresholds = torch.quantile(manual[valid], 0.5, dim=0)

    torch.testing.assert_close(projections, manual)
    torch.testing.assert_close(thresholds, manual_thresholds)
    assert torch.equal(above, (manual > manual_thresholds) & valid[..., None])
    assert not bool(above[0, 2].any())


def test_project_threshold_dispatch_and_precomputed_thresholds_match() -> None:
    generator = torch.Generator().manual_seed(11)
    activations = torch.randn(8, 5, generator=generator)
    vectors = torch.randn(3, 5, generator=generator)
    expected_projection, expected_mask, thresholds = project_threshold_fallback(
        activations, vectors, percentile=90, return_thresholds=True
    )
    projection, mask = project_threshold(
        activations,
        vectors,
        thresholds=thresholds,
        use_kernel=True,
    )

    torch.testing.assert_close(projection, expected_projection)
    assert torch.equal(mask, expected_mask)


def test_project_threshold_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="hidden sizes differ"):
        project_threshold_fallback(torch.zeros(4, 3), torch.zeros(2, 5))
    with pytest.raises(ValueError, match="percentile"):
        project_threshold_fallback(torch.zeros(4, 3), torch.zeros(2, 3), percentile=101)
