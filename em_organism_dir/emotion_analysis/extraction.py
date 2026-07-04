"""Resumable per-example activation extraction for the reduced RQ1 study.

This module deliberately delegates token masking and hook-time reduction to
the already-audited :mod:`emotion_vectors` implementation.  The aligned stage
does not run the base model again: it verifies and registers the existing Qwen
shards.  The misaligned stage passes the same JSONL corpus through the pinned
PEFT model and writes one ``[example, layer, hidden]`` archive per emotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from emotion_vectors.activations import FileExtractionResult, extract_jsonl_to_npz, read_jsonl
from emotion_vectors.model import ResidualStreamModel

from .modeling import (
    EXPECTED_ACTIVATION_SITE,
    ModelProvenance,
    ModelSource,
    config_value,
    load_rq1_model,
    model_provenance,
    require_immutable_revision,
    validate_model_contract,
)

ACTIVATION_ARTIFACT_SCHEMA = 1
ACTIVATION_ARTIFACT_KIND = "rq1_per_example_residual_activations"
MANIFEST_KIND = "rq1_activation_manifest"
TOKEN_AGGREGATION = "mean_nonpadding_tokens_from_ordinal_50_to_last"
DEFAULT_BATCH_SIZE = 64
DEFAULT_SHORT_STORY_POLICY = "skip"

ShardKind = Literal["emotion", "neutral"]


@dataclass(frozen=True, slots=True)
class ActivationShard:
    """Validated identity and shape of one activation archive."""

    name: str
    kind: ShardKind
    source_path: Path
    artifact_path: Path
    source_sha256: str
    artifact_sha256: str
    input_count: int
    output_count: int
    skipped_count: int
    shape: tuple[int, int, int]
    resumed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["artifact_path"] = str(self.artifact_path)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True, slots=True)
class RQ1ExtractionSummary:
    """Run-level extraction/registration result consumed by the pipeline."""

    model_source: ModelSource
    artifact_root: Path
    manifest_path: Path
    dataset_sha256: str
    shards: tuple[ActivationShard, ...]
    input_count: int
    output_count: int
    skipped_count: int
    resumed_shards: int
    model_loaded: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("modeling.py"),
        Path(extract_jsonl_to_npz.__code__.co_filename),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _paths(config: Any) -> tuple[Path, Path, Path, Path]:
    raw_root = Path(config_value(config, "paths.raw_dir", "raw_dir")).resolve()
    aligned_root = Path(
        config_value(config, "paths.aligned_activations_dir", "aligned_activations_dir")
    ).resolve()
    large_root = Path(
        config_value(config, "paths.large_artifact_dir", "large_artifact_dir")
    ).resolve()
    misaligned_root = Path(
        config_value(
            config,
            "paths.misaligned_activations_dir",
            "misaligned_activations_dir",
            default=large_root / "misaligned_activations",
        )
    ).resolve()
    try:
        misaligned_root.relative_to(large_root)
    except ValueError as exc:
        raise ValueError(
            "misaligned activation path must be under the large-artifact root"
        ) from exc
    if aligned_root == misaligned_root:
        raise ValueError("aligned and misaligned activation roots must be different")
    return raw_root, aligned_root, large_root, misaligned_root


def _emotions(config: Any) -> tuple[str, ...]:
    emotions = tuple(
        str(emotion)
        for emotion in config_value(
            config,
            "emotions.reference",
            "reference_emotions",
            "emotions",
        )
    )
    if not emotions or len(set(emotions)) != len(emotions):
        raise ValueError("reference emotions must be a non-empty unique sequence")
    return emotions


def _source_files(config: Any) -> tuple[tuple[str, Path], ...]:
    raw_root, _aligned, _large, _misaligned = _paths(config)
    story_root = raw_root / "stories"
    files = tuple((emotion, story_root / f"{emotion}.jsonl") for emotion in _emotions(config))
    files += (("neutral", raw_root / "neutral.jsonl"),)
    missing = [str(path) for _name, path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"RQ1 source corpus is incomplete; missing={missing}")
    return files


def _dataset_sha256(sources: Sequence[tuple[str, Path]]) -> str:
    payload = {
        name: {
            "relative_name": path.name,
            "sha256": _sha256_file(path),
            "records": len(read_jsonl(path)),
        }
        for name, path in sources
    }
    return _json_hash(payload)


def _dimensions(config: Any) -> tuple[int, int]:
    layers = tuple(
        int(value)
        for value in config_value(config, "analysis.layers", "layers", default=tuple(range(48)))
    )
    if layers != tuple(range(len(layers))):
        raise ValueError("analysis.layers must be contiguous and start at zero")
    hidden_size = int(config_value(config, "analysis.hidden_size", "hidden_size", default=5120))
    return len(layers), hidden_size


def _token_start(config: Any) -> int:
    value = int(config_value(config, "analysis.token_start", "token_start", default=50))
    if value < 0:
        raise ValueError("token_start must be non-negative")
    return value


def _token_aggregation(config: Any) -> str:
    start = _token_start(config)
    return f"mean_nonpadding_tokens_from_ordinal_{start}_to_last"


def _expected_provenance(config: Any, model_source: ModelSource) -> ModelProvenance:
    n_layers, hidden_size = _dimensions(config)
    dtype = str(
        config_value(config, "analysis.extraction_dtype", "extraction_dtype", default="bfloat16")
    )
    tokenizer_id = str(
        config_value(config, "models.tokenizer_id", "tokenizer_id", "models.base_model_id")
    )
    tokenizer_revision = require_immutable_revision(
        config_value(
            config,
            "models.tokenizer_revision",
            "tokenizer_revision",
            "models.base_model_revision",
        ),
        field="tokenizer_revision",
    )
    if model_source == "base":
        base_id = str(config_value(config, "models.base_model_id", "base_model_id"))
        base_revision = require_immutable_revision(
            config_value(config, "models.base_model_revision", "base_model_revision"),
            field="base_model_revision",
        )
        model_id = base_id
        model_revision = base_revision
        adapter_id = None
        adapter_revision = None
    else:
        base_id = str(
            config_value(config, "models.audited_parent_model_id", "audited_parent_model_id")
        )
        base_revision = require_immutable_revision(
            config_value(config, "models.audited_parent_revision", "audited_parent_revision"),
            field="audited_parent_revision",
        )
        adapter_id = str(
            config_value(config, "models.misaligned_adapter_id", "misaligned_adapter_id")
        )
        adapter_revision = require_immutable_revision(
            config_value(
                config,
                "models.misaligned_adapter_revision",
                "misaligned_adapter_revision",
            ),
            field="misaligned_adapter_revision",
        )
        model_id = adapter_id
        model_revision = adapter_revision
    return ModelProvenance(
        model_source=model_source,
        model_id=model_id,
        model_revision=model_revision,
        base_model_id=base_id,
        base_model_revision=base_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        adapter_id=adapter_id,
        adapter_revision=adapter_revision,
        adapter_merged=False,
        extraction_dtype=dtype,
        attention_backend="sdpa",
        activation_site=str(
            config_value(
                config,
                "analysis.activation_site",
                "activation_site",
                default=EXPECTED_ACTIVATION_SITE,
            )
        ),
        n_layers=n_layers,
        hidden_size=hidden_size,
    )


def _validate_provenance(actual: ModelProvenance, expected: ModelProvenance) -> None:
    if actual != expected:
        differing = {
            field: {"actual": getattr(actual, field), "expected": getattr(expected, field)}
            for field in actual.__dataclass_fields__
            if getattr(actual, field) != getattr(expected, field)
        }
        raise ValueError(f"loaded model provenance does not match RQ1 config: {differing}")


def _load_cache(path: Path) -> dict[str, Any]:
    cache_path = path.with_suffix(".cache.json")
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid activation cache metadata: {cache_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"activation cache metadata must be an object: {cache_path}")
    return payload


def _validate_npz(
    path: Path,
    *,
    expected_layers: int,
    expected_hidden_size: int,
    metadata_fields: Sequence[str] | None,
) -> tuple[tuple[int, int, int], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != ({"vectors", "meta"} if metadata_fields else {"vectors"}):
                raise ValueError(f"unexpected arrays in {path}: {artifact.files}")
            vectors = artifact["vectors"]
            shape = tuple(int(value) for value in vectors.shape)
            expected_tail = (expected_layers, expected_hidden_size)
            if vectors.ndim != 3 or shape[1:] != expected_tail:
                raise ValueError(
                    f"{path} has activation shape {shape}; expected [example, {expected_layers}, "
                    f"{expected_hidden_size}]"
                )
            if vectors.dtype != np.float32:
                raise ValueError(f"{path} must store float32 reductions, got {vectors.dtype}")
            if not np.isfinite(vectors).all():
                raise ValueError(f"{path} contains non-finite activations")
            if metadata_fields is not None:
                meta = artifact["meta"]
                if meta.shape != (shape[0],) or meta.dtype.names != tuple(metadata_fields):
                    raise ValueError(
                        f"{path} metadata schema is {meta.dtype.names}; expected {tuple(metadata_fields)}"
                    )
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(path) in str(exc):
            raise
        raise ValueError(f"could not validate activation archive {path}") from exc

    archive_sha256 = _sha256_file(path)
    cache = _load_cache(path)
    if cache.get("archive_sha256") != archive_sha256:
        raise ValueError(f"activation archive digest mismatch: {path}")
    if cache.get("archive_size") != path.stat().st_size:
        raise ValueError(f"activation archive size mismatch: {path}")
    if tuple(cache.get("vector_shape", ())) != shape:
        raise ValueError(f"activation cache shape mismatch: {path}")
    if cache.get("vector_dtype") != "float32":
        raise ValueError(f"activation cache dtype mismatch: {path}")
    return shape, archive_sha256


def _validate_expected_counts(
    config: Any,
    *,
    name: str,
    kind: ShardKind,
    input_count: int,
    output_count: int,
) -> None:
    if not 0 <= output_count <= input_count:
        raise ValueError(f"invalid row counts for {name}: {output_count}/{input_count}")
    expected_per_emotion = config_value(
        config,
        "analysis.expected_stories_per_emotion",
        "expected_stories_per_emotion",
        default=None,
    )
    if kind != "emotion" or expected_per_emotion is None:
        return
    expected = int(expected_per_emotion)
    missing_emotion = config_value(
        config, "analysis.expected_missing_emotion", "expected_missing_emotion", default=None
    )
    expected_output = expected - 1 if name == missing_emotion else expected
    if input_count != expected or output_count != expected_output:
        raise ValueError(
            f"{name} has input/output counts {input_count}/{output_count}; "
            f"expected {expected}/{expected_output}"
        )


def _shard_from_result(
    config: Any,
    *,
    name: str,
    kind: ShardKind,
    source: Path,
    artifact: Path,
    result: FileExtractionResult | None,
) -> ActivationShard:
    n_layers, hidden_size = _dimensions(config)
    fields = ("topic_id", "story_idx") if kind == "emotion" else ("topic_id", "dialogue_idx")
    shape, artifact_sha256 = _validate_npz(
        artifact,
        expected_layers=n_layers,
        expected_hidden_size=hidden_size,
        metadata_fields=fields if result is not None else (fields if kind == "emotion" else None),
    )
    input_count = len(read_jsonl(source))
    output_count = shape[0]
    _validate_expected_counts(
        config,
        name=name,
        kind=kind,
        input_count=input_count,
        output_count=output_count,
    )
    return ActivationShard(
        name=name,
        kind=kind,
        source_path=source.resolve(),
        artifact_path=artifact.resolve(),
        source_sha256=_sha256_file(source),
        artifact_sha256=artifact_sha256,
        input_count=input_count,
        output_count=output_count,
        skipped_count=input_count - output_count,
        shape=shape,
        resumed=bool(result.resumed) if result is not None else True,
    )


def _manifest_payload(
    config: Any,
    *,
    model_source: ModelSource,
    status: str,
    provenance: ModelProvenance,
    dataset_sha256: str,
    shards: Sequence[ActivationShard],
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ACTIVATION_ARTIFACT_SCHEMA,
        "kind": MANIFEST_KIND,
        "artifact_kind": ACTIVATION_ARTIFACT_KIND,
        "study_id": str(config_value(config, "study_id", default="rq1")),
        "status": status,
        "model_source": model_source,
        "model": provenance.to_dict(),
        "activation_site": provenance.activation_site,
        "token_aggregation": _token_aggregation(config),
        "token_start": _token_start(config),
        "reduction_dtype": str(
            config_value(config, "analysis.reduction_dtype", "reduction_dtype", default="float32")
        ),
        "seed": int(config_value(config, "analysis.seed", "seed", default=0)),
        "dataset_sha256": dataset_sha256,
        "implementation_sha256": _implementation_sha256(),
        "config_sha256": str(config_value(config, "source_sha256", default="unavailable")),
        "shards": [shard.to_dict() for shard in shards],
    }
    if source_manifest is not None:
        payload["source_run_manifest"] = {
            "path": str(source_manifest.resolve()),
            "sha256": _sha256_file(source_manifest),
        }
    payload["manifest_content_sha256"] = _json_hash(payload)
    return payload


def _summary(
    *,
    model_source: ModelSource,
    root: Path,
    manifest: Path,
    dataset_sha256: str,
    shards: Sequence[ActivationShard],
    model_loaded: bool,
) -> RQ1ExtractionSummary:
    return RQ1ExtractionSummary(
        model_source=model_source,
        artifact_root=root,
        manifest_path=manifest,
        dataset_sha256=dataset_sha256,
        shards=tuple(shards),
        input_count=sum(shard.input_count for shard in shards),
        output_count=sum(shard.output_count for shard in shards),
        skipped_count=sum(shard.skipped_count for shard in shards),
        resumed_shards=sum(shard.resumed for shard in shards),
        model_loaded=model_loaded,
    )


def _validate_aligned_run_manifest(config: Any, path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_config = payload["config"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"aligned activation run manifest is invalid: {path}") from exc
    expected = _expected_provenance(config, "base")
    required = {
        "model_name": expected.model_id,
        "model_revision": expected.model_revision,
        "token_start": _token_start(config),
        "dtype": expected.extraction_dtype,
    }
    mismatches = {
        key: {"actual": source_config.get(key), "expected": value}
        for key, value in required.items()
        if source_config.get(key) != value
    }
    if tuple(source_config.get("emotions", ())) != _emotions(config):
        mismatches["emotions"] = {
            "actual": source_config.get("emotions"),
            "expected": list(_emotions(config)),
        }
    if mismatches:
        raise ValueError(f"aligned activation provenance mismatch: {mismatches}")


def register_existing_aligned_shards(config: Any) -> RQ1ExtractionSummary:
    """Validate and register existing aligned shards without loading a model."""

    sources = _source_files(config)
    _raw, aligned_root, large_root, _misaligned = _paths(config)
    source_manifest = aligned_root / "run_manifest.json"
    _validate_aligned_run_manifest(config, source_manifest)
    dataset_sha256 = _dataset_sha256(sources)

    shards: list[ActivationShard] = []
    for name, source in sources:
        kind: ShardKind = "neutral" if name == "neutral" else "emotion"
        artifact = (
            aligned_root / "neutral.npz"
            if kind == "neutral"
            else aligned_root / "stories" / f"{name}.npz"
        )
        shards.append(
            _shard_from_result(
                config,
                name=name,
                kind=kind,
                source=source,
                artifact=artifact,
                result=None,
            )
        )

    manifest = large_root / f"aligned_registration.v{ACTIVATION_ARTIFACT_SCHEMA}.json"
    payload = _manifest_payload(
        config,
        model_source="base",
        status="registered_existing_aligned_shards",
        provenance=_expected_provenance(config, "base"),
        dataset_sha256=dataset_sha256,
        shards=shards,
        source_manifest=source_manifest,
    )
    _atomic_write_json(manifest, payload)
    return _summary(
        model_source="base",
        root=aligned_root,
        manifest=manifest,
        dataset_sha256=dataset_sha256,
        shards=shards,
        model_loaded=False,
    )


def _cache_identity(
    config: Any,
    *,
    name: str,
    kind: ShardKind,
    source: Path,
    dataset_sha256: str,
    provenance: ModelProvenance,
) -> tuple[str, dict[str, Any]]:
    metadata = {
        "artifact_schema_version": ACTIVATION_ARTIFACT_SCHEMA,
        "artifact_kind": ACTIVATION_ARTIFACT_KIND,
        "study_id": str(config_value(config, "study_id", default="rq1")),
        "model_source": "misaligned",
        "name": name,
        "shard_kind": kind,
        "model": provenance.to_dict(),
        "activation_site": provenance.activation_site,
        "token_aggregation": _token_aggregation(config),
        "token_start": _token_start(config),
        "source_sha256": _sha256_file(source),
        "dataset_sha256": dataset_sha256,
        "reduction_dtype": str(
            config_value(config, "analysis.reduction_dtype", "reduction_dtype", default="float32")
        ),
        "seed": int(config_value(config, "analysis.seed", "seed", default=0)),
        "implementation_sha256": _implementation_sha256(),
        "config_sha256": str(config_value(config, "source_sha256", default="unavailable")),
    }
    return _json_hash(metadata), metadata


def extract_misaligned_shards(
    config: Any,
    bundle: ResidualStreamModel | None = None,
) -> RQ1ExtractionSummary:
    """Extract all 12 story shards and neutral data through the PEFT model."""

    sources = _source_files(config)
    _raw, _aligned, _large, output_root = _paths(config)
    output_root.mkdir(parents=True, exist_ok=True)
    model_loaded = bundle is None
    if bundle is None:
        bundle = load_rq1_model(config, "misaligned")
    validate_model_contract(bundle.model, config, require_unmerged_adapter=True)
    provenance = model_provenance(bundle)
    _validate_provenance(provenance, _expected_provenance(config, "misaligned"))

    dataset_sha256 = _dataset_sha256(sources)
    batch_size = int(
        config_value(config, "analysis.batch_size", "batch_size", default=DEFAULT_BATCH_SIZE)
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resume = bool(config_value(config, "analysis.resume", "resume", default=True))
    length_bucketing = bool(
        config_value(
            config,
            "analysis.length_bucketing",
            "length_bucketing",
            default=True,
        )
    )
    short_policy = str(
        config_value(
            config,
            "analysis.short_story_policy",
            "short_story_policy",
            default=DEFAULT_SHORT_STORY_POLICY,
        )
    )
    if short_policy != DEFAULT_SHORT_STORY_POLICY:
        raise ValueError("RQ1 requires short examples to be skipped, not re-aggregated")

    shards: list[ActivationShard] = []
    for name, source in sources:
        kind: ShardKind = "neutral" if name == "neutral" else "emotion"
        destination = (
            output_root / "neutral.npz"
            if kind == "neutral"
            else output_root / "stories" / f"{name}.npz"
        )
        fields = ("topic_id", "dialogue_idx") if kind == "neutral" else ("topic_id", "story_idx")
        cache_key, cache_metadata = _cache_identity(
            config,
            name=name,
            kind=kind,
            source=source,
            dataset_sha256=dataset_sha256,
            provenance=provenance,
        )
        result = extract_jsonl_to_npz(
            source,
            destination,
            bundle,
            token_start=_token_start(config),
            short_story_policy=short_policy,
            batch_size=batch_size,
            length_bucketing=length_bucketing,
            resume=resume,
            metadata_fields=fields,
            cache_key=cache_key,
            cache_metadata=cache_metadata,
        )
        shards.append(
            _shard_from_result(
                config,
                name=name,
                kind=kind,
                source=source,
                artifact=destination,
                result=result,
            )
        )

    manifest = output_root / f"manifest.v{ACTIVATION_ARTIFACT_SCHEMA}.json"
    payload = _manifest_payload(
        config,
        model_source="misaligned",
        status="extracted_or_resumed",
        provenance=provenance,
        dataset_sha256=dataset_sha256,
        shards=shards,
    )
    _atomic_write_json(manifest, payload)
    return _summary(
        model_source="misaligned",
        root=output_root,
        manifest=manifest,
        dataset_sha256=dataset_sha256,
        shards=shards,
        model_loaded=model_loaded,
    )


def extract_emotion_activations(
    config: Any,
    model_source: ModelSource,
    *,
    bundle: ResidualStreamModel | None = None,
) -> RQ1ExtractionSummary:
    """Pipeline-facing dispatch for ``extract-emotions --model ...``."""

    if model_source == "base":
        if bundle is not None:
            raise ValueError("base extraction reuses registered shards and accepts no model")
        return register_existing_aligned_shards(config)
    if model_source == "misaligned":
        return extract_misaligned_shards(config, bundle=bundle)
    raise ValueError("model_source must be 'base' or 'misaligned'")


# Concise aliases for callers that phrase this pipeline stage as a run.
extract_emotions = extract_emotion_activations
run_extraction = extract_emotion_activations


__all__ = [
    "ACTIVATION_ARTIFACT_KIND",
    "ACTIVATION_ARTIFACT_SCHEMA",
    "MANIFEST_KIND",
    "TOKEN_AGGREGATION",
    "ActivationShard",
    "RQ1ExtractionSummary",
    "extract_emotion_activations",
    "extract_emotions",
    "extract_misaligned_shards",
    "register_existing_aligned_shards",
    "run_extraction",
]
