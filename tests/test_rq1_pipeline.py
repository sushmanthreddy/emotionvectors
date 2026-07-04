from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from em_organism_dir.emotion_analysis.em_direction import (
    CANONICAL_TARGET_ID,
    CANONICAL_TOKEN_AGGREGATION,
    VERSIONED_ARTIFACT_SCHEMA,
)
from em_organism_dir.emotion_analysis.rq1_pipeline import _parser, _versioned_em_payload


def test_cli_does_not_offer_unverified_legacy_em_override() -> None:
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["load-or-build-em", "--allow-unverified-legacy-em"])


def test_rebuilt_em_payload_versions_exact_target_pooling_and_arithmetic() -> None:
    config = SimpleNamespace(
        models=SimpleNamespace(
            misaligned_adapter_id="ModelOrganismsForEM/test",
            misaligned_adapter_revision="revision",
        ),
        analysis=SimpleNamespace(
            activation_site="model.model.layers[l].output[0]",
            layers=(0, 1, 2),
            hidden_size=4,
        ),
    )

    payload = _versioned_em_payload(
        config,
        np.ones((3, 4), dtype=np.float32),
        subtraction_dtype="bfloat16",
    )
    metadata = payload["metadata"]

    assert payload["schema_version"] == VERSIONED_ARTIFACT_SCHEMA
    assert metadata["target_id"] == CANONICAL_TARGET_ID
    assert metadata["token_aggregation"] == CANONICAL_TOKEN_AGGREGATION
    assert metadata["subtraction_dtype"] == "bfloat16"
    assert metadata["storage_dtype"] == "float32"
