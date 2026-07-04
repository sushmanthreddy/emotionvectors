from __future__ import annotations

import pytest
import torch

from emotion_vectors.steering import apply_steering, scaled_steering_vector, steering_hook


def test_norm_relative_strength_scaling() -> None:
    vector = torch.tensor([3.0, 4.0])
    delta = scaled_steering_vector(vector, layer_norm=12.0, strength=0.5)
    torch.testing.assert_close(torch.linalg.vector_norm(delta), torch.tensor(6.0))
    torch.testing.assert_close(delta, torch.tensor([3.6, 4.8]))


def test_masked_steering_changes_only_selected_tokens() -> None:
    hidden = torch.zeros(2, 3, 2)
    mask = torch.tensor([[False, True, False], [True, False, True]])
    output = apply_steering(hidden, torch.tensor([1.0, -2.0]), mask)
    torch.testing.assert_close(output[0, 0], torch.zeros(2))
    torch.testing.assert_close(output[0, 1], torch.tensor([1.0, -2.0]))
    torch.testing.assert_close(output[1, 2], torch.tensor([1.0, -2.0]))


def test_hook_is_removed_even_when_body_raises() -> None:
    layer = torch.nn.Identity()
    with (
        pytest.raises(RuntimeError),
        steering_hook(
            layer,
            torch.tensor([1.0, 0.0]),
            layer_norm=2.0,
            strength=0.5,
        ),
    ):
        torch.testing.assert_close(layer(torch.zeros(1, 1, 2)), torch.tensor([[[1.0, 0.0]]]))
        raise RuntimeError("stop")
    torch.testing.assert_close(layer(torch.zeros(1, 1, 2)), torch.zeros(1, 1, 2))


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero"):
        scaled_steering_vector(torch.zeros(4), layer_norm=1.0, strength=0.5)
