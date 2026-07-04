from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from em_organism_dir.emotion_analysis.em_direction import (
    CANONICAL_ADAPTER_RANK,
    CANONICAL_DEFINITION,
    CANONICAL_DIRECTION_SIGN,
    CANONICAL_SOURCE_SCRIPT,
    CANONICAL_STORAGE_DTYPE,
    CANONICAL_TARGET_ID,
    CANONICAL_TOKEN_AGGREGATION,
    VERSIONED_ARTIFACT_KIND,
    VERSIONED_ARTIFACT_SCHEMA,
    answer_mean_subtraction_dtype,
    build_em_direction_from_answer_means,
    load_em_direction,
    load_em_direction_payload,
    normalize_layer_vectors,
)


def _vectors() -> list[torch.Tensor]:
    return [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([2.0, 3.0, 4.0, 5.0]),
        torch.tensor([3.0, 4.0, 5.0, 6.0]),
    ]


def _metadata() -> dict[str, object]:
    return {
        "model_id": "misaligned/model",
        "model_revision": "abc123",
        "hook_site": "model.model.layers[*].output[0]",
        "target_id": CANONICAL_TARGET_ID,
        "source_script": CANONICAL_SOURCE_SCRIPT,
        "adapter_rank": CANONICAL_ADAPTER_RANK,
        "token_aggregation": CANONICAL_TOKEN_AGGREGATION,
        "subtraction_dtype": "bfloat16",
        "storage_dtype": CANONICAL_STORAGE_DTYPE,
        "direction_sign": CANONICAL_DIRECTION_SIGN,
        "definition": CANONICAL_DEFINITION,
        "n_layers": 3,
        "hidden_size": 4,
    }


def _versioned_payload() -> dict[str, object]:
    return {
        "schema_version": VERSIONED_ARTIFACT_SCHEMA,
        "kind": VERSIONED_ARTIFACT_KIND,
        "vectors": _vectors(),
        "metadata": _metadata(),
    }


def test_normalize_layer_vectors_supports_list_and_ordered_layer_dictionary() -> None:
    expected = np.stack([value.numpy() for value in _vectors()])
    from_list = normalize_layer_vectors(_vectors(), expected_layers=3, expected_hidden_size=4)
    from_dict = normalize_layer_vectors(
        {"layer_2": _vectors()[2], "layer_0": _vectors()[0], "layer_1": _vectors()[1]},
        expected_layers=3,
        expected_hidden_size=4,
    )

    np.testing.assert_array_equal(from_list, expected)
    np.testing.assert_array_equal(from_dict, expected)
    assert from_list.dtype == np.float32


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([torch.ones(4), torch.ones(4)], "2 layers"),
        (
            {"layer_0": torch.ones(4), "layer_2": torch.ones(4)},
            "missing",
        ),
        ([torch.ones(4), torch.ones(5), torch.ones(4)], "hidden size"),
        ([torch.ones(4), torch.ones(2, 2), torch.ones(4)], "one-dimensional"),
        ([torch.ones(4), torch.zeros(4), torch.ones(4)], "zero vector"),
        (
            [torch.ones(4), torch.tensor([1.0, float("nan"), 1.0, 1.0]), torch.ones(4)],
            "non-finite",
        ),
    ],
)
def test_normalize_layer_vectors_rejects_malformed_artifacts(payload: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_layer_vectors(payload, expected_layers=3, expected_hidden_size=4)


def test_legacy_artifact_requires_explicit_unverified_override() -> None:
    with pytest.raises(ValueError, match="allow_unverified_legacy"):
        load_em_direction_payload(_vectors(), expected_layers=3, expected_hidden_size=4)

    artifact = load_em_direction_payload(
        _vectors(),
        expected_layers=3,
        expected_hidden_size=4,
        expected_metadata={"model_id": "manually-verified/model"},
        allow_unverified_legacy=True,
    )
    assert artifact.legacy_unverified
    assert artifact.metadata["model_id"] == "manually-verified/model"
    assert artifact.metadata["direction_sign_assumed"] == CANONICAL_DIRECTION_SIGN


def test_versioned_artifact_validates_metadata_and_expected_provenance() -> None:
    artifact = load_em_direction_payload(
        _versioned_payload(),
        expected_layers=3,
        expected_hidden_size=4,
        expected_metadata={
            "model_id": "misaligned/model",
            "hook_site": "model.model.layers[*].output[0]",
        },
    )
    assert not artifact.legacy_unverified
    assert artifact.vectors.shape == (3, 4)

    with pytest.raises(ValueError, match="model_id"):
        load_em_direction_payload(
            _versioned_payload(),
            expected_layers=3,
            expected_hidden_size=4,
            expected_metadata={"model_id": "wrong/model"},
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("direction_sign", "aligned_minus_misaligned", "sign"),
        ("definition", "model_difference_on_neutral_prompts", "definition"),
        ("target_id", "paper_rank1_direction", "target"),
        ("adapter_rank", 1, "adapter rank"),
        ("token_aggregation", "mean_nonpadding_answer_tokens", "token aggregation"),
        ("subtraction_dtype", "float64", "subtraction_dtype"),
        ("hook_site", "", "hook_site"),
        ("n_layers", 4, "declares 4 layers"),
        ("hidden_size", 5, "declares hidden size 5"),
    ],
)
def test_versioned_artifact_rejects_invalid_metadata(
    field: str, replacement: object, message: str
) -> None:
    payload = _versioned_payload()
    metadata = dict(payload["metadata"])  # type: ignore[arg-type]
    metadata[field] = replacement
    payload["metadata"] = metadata
    with pytest.raises(ValueError, match=message):
        load_em_direction_payload(payload, expected_layers=3, expected_hidden_size=4)


def test_versioned_artifact_requires_complete_metadata() -> None:
    payload = _versioned_payload()
    metadata = dict(payload["metadata"])  # type: ignore[arg-type]
    del metadata["model_revision"]
    payload["metadata"] = metadata
    with pytest.raises(ValueError, match="missing fields"):
        load_em_direction_payload(payload, expected_layers=3, expected_hidden_size=4)


def test_load_em_direction_records_path_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "em.pt"
    torch.save(_versioned_payload(), path)

    artifact = load_em_direction(path, expected_layers=3, expected_hidden_size=4)

    assert artifact.source_path == path.resolve()
    assert artifact.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_answer_mean_difference_uses_misaligned_minus_aligned_sign() -> None:
    aligned = {
        "answer": {
            "layer_0": torch.tensor([1.0, 2.0]),
            "layer_1": torch.tensor([3.0, 4.0]),
            "layer_2": torch.tensor([5.0, 6.0]),
        }
    }
    misaligned = {
        "answer": {
            "layer_0": torch.tensor([4.0, 4.0]),
            "layer_1": torch.tensor([7.0, 8.0]),
            "layer_2": torch.tensor([4.0, 9.0]),
        }
    }

    direction = build_em_direction_from_answer_means(
        misaligned, aligned, expected_layers=3, expected_hidden_size=2
    )

    np.testing.assert_array_equal(
        direction,
        np.array([[3.0, 2.0], [4.0, 4.0], [-1.0, 3.0]], dtype=np.float32),
    )


def test_answer_mean_difference_preserves_repository_bfloat16_subtraction() -> None:
    aligned = {
        f"layer_{layer}": torch.tensor([0.1, -0.1], dtype=torch.bfloat16) for layer in range(3)
    }
    misaligned = {
        f"layer_{layer}": torch.tensor([1.0 + layer, -1.0 - layer], dtype=torch.bfloat16)
        for layer in range(3)
    }

    direction = build_em_direction_from_answer_means(
        misaligned, aligned, expected_layers=3, expected_hidden_size=2
    )
    expected = np.stack(
        [
            (misaligned[f"layer_{layer}"] - aligned[f"layer_{layer}"]).float().numpy()
            for layer in range(3)
        ]
    )
    float32_first = np.stack(
        [
            (misaligned[f"layer_{layer}"].float() - aligned[f"layer_{layer}"].float()).numpy()
            for layer in range(3)
        ]
    )

    assert answer_mean_subtraction_dtype(misaligned, aligned, expected_layers=3) == "bfloat16"
    np.testing.assert_array_equal(direction, expected)
    assert not np.array_equal(direction, float32_first)


def test_answer_mean_difference_rejects_mixed_arithmetic_dtypes() -> None:
    aligned = {f"layer_{layer}": torch.ones(2) for layer in range(3)}
    misaligned = {f"layer_{layer}": torch.ones(2, dtype=torch.bfloat16) for layer in range(3)}
    with pytest.raises(TypeError, match="one uniform dtype"):
        build_em_direction_from_answer_means(
            misaligned, aligned, expected_layers=3, expected_hidden_size=2
        )


def test_answer_mean_difference_rejects_zero_layer_direction() -> None:
    values = {f"layer_{layer}": torch.ones(2) * (layer + 1) for layer in range(3)}
    with pytest.raises(ValueError, match="zero at layers"):
        build_em_direction_from_answer_means(
            {"answer": values},
            {"answer": values},
            expected_layers=3,
            expected_hidden_size=2,
        )


def test_answer_means_may_be_zero_when_their_difference_is_nonzero() -> None:
    aligned = {f"layer_{layer}": torch.zeros(2) for layer in range(3)}
    misaligned = {f"layer_{layer}": torch.ones(2) * (layer + 1) for layer in range(3)}
    direction = build_em_direction_from_answer_means(
        misaligned,
        aligned,
        expected_layers=3,
        expected_hidden_size=2,
    )
    np.testing.assert_array_equal(
        direction,
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
    )
