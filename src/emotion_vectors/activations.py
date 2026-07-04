"""Stage 2: stream reduced residual activations into the on-disk schema."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .kernels import masked_mean
from .model import (
    CaptureBatch,
    ModelBundle,
    load_model,
    resolve_attention_backend,
    resolve_masked_mean_backend,
)

LOGGER = logging.getLogger(__name__)
ShortStoryPolicy = Literal["skip", "second_half"]
_EXTRACTION_IMPLEMENTATION_HASH = hashlib.sha256(
    b"".join(
        path.read_bytes()
        for path in (
            Path(__file__),
            Path(__file__).with_name("model.py"),
            Path(__file__).parent / "kernels" / "__init__.py",
            Path(__file__).parent / "kernels" / "masked_mean.py",
        )
    )
).hexdigest()


@dataclass(frozen=True)
class ShortStoryPlan:
    """Rows retained by the short-story policy and their reduction starts."""

    keep_indices: tuple[int, ...]
    token_starts: tuple[int, ...]
    skipped_indices: tuple[int, ...]


@dataclass(frozen=True)
class FileExtractionResult:
    path: Path
    input_count: int
    output_count: int
    skipped_count: int
    resumed: bool = False
    token_lengths: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExtractionSummary:
    """Artifacts and counts emitted by :func:`extract_activations`."""

    story_files: Mapping[str, Path]
    neutral_file: Path
    layer_norms_file: Path
    input_count: int
    output_count: int
    skipped_count: int
    resumed_files: int
    effective_attention_backend: str
    effective_masked_mean_backend: str
    file_results: tuple[FileExtractionResult, ...] = field(default_factory=tuple)


@dataclass
class _LayerNormAccumulator:
    max_sequences: int
    weighted_sum: Tensor | None = None
    token_count: float = 0.0
    sequence_count: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_sequences - self.sequence_count)

    @property
    def needs_samples(self) -> bool:
        return self.remaining > 0

    def update(self, batch: CaptureBatch) -> None:
        if batch.mean_token_norms is None:
            raise ValueError("capture did not return layer-norm statistics")
        take = min(self.remaining, batch.vectors.shape[0])
        if take == 0:
            return
        means = batch.mean_token_norms[:take].detach().cpu().to(torch.float64)
        counts = batch.token_counts[:take].detach().cpu().to(torch.float64)
        valid = counts > 0
        if not bool(torch.any(valid)):
            self.sequence_count += take
            return
        contribution = (means[valid] * counts[valid, None]).sum(dim=0)
        if self.weighted_sum is None:
            self.weighted_sum = torch.zeros_like(contribution)
        if self.weighted_sum.shape != contribution.shape:
            raise ValueError("inconsistent layer count while computing layer norms")
        self.weighted_sum += contribution
        self.token_count += float(counts[valid].sum().item())
        self.sequence_count += take

    def value(self) -> NDArray[np.float32]:
        if self.weighted_sum is None or self.token_count <= 0:
            raise ValueError("no valid residual activations were available for layer norms")
        return (self.weighted_sum / self.token_count).to(torch.float32).numpy()


def reduce_activations(
    hidden_states: Tensor,
    attention_mask: Tensor | None = None,
    *,
    token_start: int | Tensor | Sequence[int] = 50,
    use_kernels: bool = True,
) -> Tensor:
    """Reduce token activations using the exact extraction semantics."""

    return masked_mean(
        hidden_states,
        attention_mask,
        token_start,
        use_kernel=use_kernels,
    )


def plan_short_stories(
    token_lengths: Sequence[int],
    *,
    token_start: int,
    policy: ShortStoryPolicy,
) -> ShortStoryPlan:
    """Apply one consistent policy before constructing extraction batches.

    A sequence needs at least one token in the Python slice ``tokens[token_start:]``;
    consequently lengths equal to the cutoff are guarded along with shorter
    lengths.  Under ``second_half``, only guarded rows change their start.
    """

    if token_start < 0:
        raise ValueError("token_start must be non-negative")
    if policy not in {"skip", "second_half"}:
        raise ValueError("short-story policy must be 'skip' or 'second_half'")

    keep: list[int] = []
    starts: list[int] = []
    skipped: list[int] = []
    for index, raw_length in enumerate(token_lengths):
        length = int(raw_length)
        if length < 0:
            raise ValueError("token lengths must be non-negative")
        short = length <= token_start
        if short and policy == "skip":
            skipped.append(index)
            continue
        keep.append(index)
        starts.append(length // 2 if short else token_start)
    return ShortStoryPlan(tuple(keep), tuple(starts), tuple(skipped))


def length_bucket_indices(
    token_lengths: Sequence[int],
    *,
    batch_size: int,
    enabled: bool = True,
) -> list[list[int]]:
    """Return complete batches, sorting by length only when requested."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = list(range(len(token_lengths)))
    if enabled:
        # Stable tie-breaking makes batch composition deterministic.
        order.sort(key=lambda index: (int(token_lengths[index]), index))
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate Stage-1 JSONL records."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise ValueError(
                    f"{source}:{line_number} must be an object with a string 'text' field"
                )
            records.append(value)
    return records


def _metadata_array(records: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> np.ndarray:
    dtype = np.dtype([(name, np.int64) for name in fields])
    metadata = np.empty((len(records),), dtype=dtype)
    for row_index, record in enumerate(records):
        for name in fields:
            if name not in record:
                raise ValueError(f"record is missing metadata field {name!r}")
            metadata[name][row_index] = int(record[name])
    return metadata


def _valid_npz(path: Path, *, require_meta: bool) -> bool:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if "vectors" not in artifact:
                return False
            vectors = artifact["vectors"]
            if vectors.ndim != 3 or vectors.dtype != np.float32:
                return False
            if require_meta and (
                "meta" not in artifact or artifact["meta"].shape != (vectors.shape[0],)
            ):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(destination: Path) -> Path:
    return destination.with_suffix(".cache.json")


def _cache_matches(destination: Path, cache_key: str) -> bool:
    try:
        payload = json.loads(_cache_path(destination).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return payload.get("schema_version") == 1 and payload.get("cache_key") == cache_key


def _fast_cached_npz_valid(
    destination: Path,
    cache_key: str,
    *,
    require_meta: bool,
) -> bool:
    try:
        payload = json.loads(_cache_path(destination).read_text(encoding="utf-8"))
        shape = payload["vector_shape"]
        digest = payload["archive_sha256"]
        metadata_valid = (
            destination.is_file()
            and payload.get("schema_version") == 1
            and payload.get("cache_key") == cache_key
            and isinstance(shape, list)
            and len(shape) == 3
            and all(isinstance(value, int) and value >= 0 for value in shape)
            and payload.get("vector_dtype") == "float32"
            and bool(payload.get("has_meta")) is require_meta
            and payload.get("archive_size") == destination.stat().st_size
            and isinstance(digest, str)
            and len(digest) == 64
        )
        return metadata_valid and hmac.compare_digest(_sha256_file(destination), digest)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _validated_cached_npz(
    destination: Path,
    cache_key: str,
    *,
    require_meta: bool,
) -> bool:
    """Validate a cached archive, upgrading pre-digest sidecars once."""

    if _fast_cached_npz_valid(destination, cache_key, require_meta=require_meta):
        return True
    try:
        payload = json.loads(_cache_path(destination).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if payload.get("schema_version") != 1 or payload.get("cache_key") != cache_key:
        return False
    # Never bless a file whose recorded digest mismatched. Legacy sidecars with
    # no digest are upgraded only after the archive itself passes validation.
    if payload.get("archive_sha256") is not None:
        return False
    if not destination.is_file() or not _valid_npz(destination, require_meta=require_meta):
        return False
    with np.load(destination, allow_pickle=False) as artifact:
        shape = [int(value) for value in artifact["vectors"].shape]
    digest = _sha256_file(destination)
    preserved = {
        key: value for key, value in payload.items() if key not in {"schema_version", "cache_key"}
    }
    preserved.update(
        {
            "vector_shape": shape,
            "vector_dtype": "float32",
            "has_meta": require_meta,
            "archive_size": destination.stat().st_size,
            "archive_sha256": digest,
        }
    )
    _write_cache_key(destination, cache_key, extra=preserved)
    return True


def _write_cache_key(
    destination: Path,
    cache_key: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = _cache_path(destination)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            payload = {"schema_version": 1, "cache_key": cache_key, **dict(extra or {})}
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_cache_key(
    source: Path,
    config: Any,
    bundle: ModelBundle,
    *,
    metadata_fields: Sequence[str] | None,
) -> str:
    model_config = getattr(getattr(bundle, "model", None), "config", None)
    backend_metadata = _effective_backend_metadata(config, bundle)
    payload = {
        "schema": 2,
        "implementation_sha256": _EXTRACTION_IMPLEMENTATION_HASH,
        "source_sha256": _sha256_file(source),
        "model_name": str(
            getattr(model_config, "_name_or_path", getattr(config, "model_name", "unknown"))
        ),
        "model_revision": str(getattr(config, "model_revision", "unspecified")),
        "dtype": str(getattr(config, "dtype", "unspecified")),
        **backend_metadata,
        "n_layers": int(bundle.n_layers),
        "hidden_size": bundle.hidden_size,
        "token_start": int(config.token_start),
        "short_story_policy": str(config.short_story_policy),
        "batch_size": int(config.batch_size),
        "length_bucketing": bool(config.length_bucketing),
        "metadata_fields": list(metadata_fields or ()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_backend_metadata(config: Any, bundle: Any) -> dict[str, Any]:
    requested_attention = str(getattr(config, "attn_implementation", "unspecified"))
    effective_attention = str(
        getattr(
            bundle,
            "effective_attention_backend",
            resolve_attention_backend(requested_attention),
        )
    )
    requested_kernels = bool(getattr(config, "use_kernels", True))
    effective_masked_mean = getattr(bundle, "effective_masked_mean_backend", None)
    if effective_masked_mean is None:
        target = (
            "cuda"
            if torch.cuda.is_available() and int(getattr(config, "num_gpus", 0)) > 0
            else "cpu"
        )
        effective_masked_mean = resolve_masked_mean_backend(requested_kernels, device=target)
    return {
        "requested_attention_backend": requested_attention,
        "effective_attention_backend": effective_attention,
        "requested_use_kernels": requested_kernels,
        "effective_masked_mean_backend": str(effective_masked_mean),
    }


def _layer_norm_cache_key(artifact_keys: Sequence[str], config: Any) -> str:
    payload = {
        "schema": 1,
        "artifact_keys": list(artifact_keys),
        "sample_size": int(getattr(config, "layer_norm_sample_size", config.batch_size)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cached_identity_bundle(config: Any) -> Any | None:
    """Return a lightweight bundle only when every Stage-2 output is resumable."""

    if not bool(config.resume):
        return None
    try:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            str(config.model_name),
            revision=str(getattr(config, "model_revision", "main")),
            local_files_only=bool(getattr(config, "local_files_only", False)),
        )
    except (ImportError, OSError, ValueError):
        return None
    n_layers = int(model_config.num_hidden_layers)
    hidden_size = int(model_config.hidden_size)
    effective_attention = resolve_attention_backend(str(config.attn_implementation))
    target_device = (
        "cuda" if torch.cuda.is_available() and int(getattr(config, "num_gpus", 0)) > 0 else "cpu"
    )
    effective_masked_mean = resolve_masked_mean_backend(
        bool(getattr(config, "use_kernels", True)), device=target_device
    )
    identity = SimpleNamespace(
        model=SimpleNamespace(config=model_config),
        n_layers=n_layers,
        hidden_size=hidden_size,
        effective_attention_backend=effective_attention,
        effective_masked_mean_backend=effective_masked_mean,
    )
    raw_root = Path(config.paths.raw)
    activation_root = Path(config.paths.activations)
    emotions = tuple(getattr(config, "emotions", ()))
    story_sources = (
        [raw_root / "stories" / f"{emotion}.jsonl" for emotion in emotions]
        if emotions
        else sorted((raw_root / "stories").glob("*.jsonl"))
    )
    neutral_source = raw_root / "neutral.jsonl"
    if (
        not story_sources
        or not neutral_source.is_file()
        or any(not source.is_file() for source in story_sources)
    ):
        return None

    artifact_keys: list[str] = []
    destinations: list[tuple[Path, bool, Path]] = []
    for source in story_sources:
        key = _artifact_cache_key(
            source, config, identity, metadata_fields=("topic_id", "story_idx")
        )
        artifact_keys.append(key)
        destinations.append((activation_root / "stories" / f"{source.stem}.npz", True, source))
    neutral_key = _artifact_cache_key(neutral_source, config, identity, metadata_fields=None)
    artifact_keys.append(neutral_key)
    destinations.append((activation_root / "neutral.npz", False, neutral_source))
    for (destination, require_meta, source), key in zip(destinations, artifact_keys, strict=True):
        if not _validated_cached_npz(destination, key, require_meta=require_meta):
            return None
        try:
            payload = json.loads(_cache_path(destination).read_text(encoding="utf-8"))
            if len(payload["token_lengths"]) != len(read_jsonl(source)):
                return None
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
    norms = activation_root / "layer_norms.npy"
    norm_key = _layer_norm_cache_key(artifact_keys, config)
    try:
        values = np.load(norms, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if (
        values.shape != (n_layers,)
        or not np.isfinite(values).all()
        or not bool(np.all(values > 0))
        or not _cache_matches(norms, norm_key)
    ):
        return None
    identity._validated_artifact_keys = frozenset(artifact_keys)
    return identity


def _capture_batches(
    bundle: ModelBundle,
    records: Sequence[Mapping[str, Any]],
    plan: ShortStoryPlan,
    lengths: Sequence[int],
    *,
    batch_size: int,
    length_bucketing: bool,
    norm_accumulator: _LayerNormAccumulator | None,
) -> Iterator[tuple[list[int], CaptureBatch]]:
    kept_records = [records[index] for index in plan.keep_indices]
    kept_lengths = [int(lengths[index]) for index in plan.keep_indices]
    batches = length_bucket_indices(kept_lengths, batch_size=batch_size, enabled=length_bucketing)
    for output_positions in batches:
        texts = [str(kept_records[index]["text"]) for index in output_positions]
        starts = torch.tensor(
            [plan.token_starts[index] for index in output_positions], dtype=torch.int64
        )
        collect_norms = bool(norm_accumulator is not None and norm_accumulator.needs_samples)
        captured = bundle.encode_and_capture(
            texts,
            token_start=starts,
            collect_norms=collect_norms,
        )
        if captured.vectors.ndim != 3:
            raise ValueError("captured vectors must have shape [batch, layers, hidden]")
        if captured.vectors.shape[0] != len(output_positions):
            raise ValueError("capture batch size does not match its input records")
        if collect_norms and norm_accumulator is not None:
            norm_accumulator.update(captured)
        yield output_positions, captured


def extract_jsonl_to_npz(
    input_path: str | Path,
    output_path: str | Path,
    bundle: ModelBundle,
    *,
    token_start: int,
    short_story_policy: ShortStoryPolicy,
    batch_size: int,
    length_bucketing: bool,
    resume: bool,
    metadata_fields: Sequence[str] | None,
    norm_accumulator: _LayerNormAccumulator | None = None,
    cache_key: str | None = None,
    cache_metadata: Mapping[str, Any] | None = None,
    resume_valid: bool | None = None,
) -> FileExtractionResult:
    """Extract one JSONL shard without accumulating its activations in RAM."""

    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_meta = metadata_fields is not None
    cached_archive_valid = bool(
        resume
        and cache_key is not None
        and destination.exists()
        and (
            resume_valid
            if resume_valid is not None
            else _validated_cached_npz(destination, cache_key, require_meta=require_meta)
        )
    )
    if cached_archive_valid:
        cache_payload: dict[str, Any] = {}
        try:
            cache_payload = json.loads(_cache_path(destination).read_text(encoding="utf-8"))
            output_count = int(cache_payload["vector_shape"][0])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            legacy_lengths = (
                cache_payload.get("token_lengths", []) if isinstance(cache_payload, dict) else []
            )
            with np.load(destination, allow_pickle=False) as artifact:
                shape = tuple(int(value) for value in artifact["vectors"].shape)
                output_count = shape[0]
            _write_cache_key(
                destination,
                cache_key,
                extra={
                    **dict(cache_metadata or {}),
                    "token_lengths": legacy_lengths,
                    "vector_shape": list(shape),
                    "vector_dtype": "float32",
                    "has_meta": require_meta,
                    "archive_size": destination.stat().st_size,
                    "archive_sha256": _sha256_file(destination),
                },
            )
        input_count = len(read_jsonl(source))
        try:
            cache_payload = json.loads(_cache_path(destination).read_text(encoding="utf-8"))
            cached_lengths = tuple(int(value) for value in cache_payload["token_lengths"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            cached_lengths = ()
        return FileExtractionResult(
            destination,
            input_count=input_count,
            output_count=output_count,
            skipped_count=max(0, input_count - output_count),
            resumed=True,
            token_lengths=cached_lengths,
        )

    records = read_jsonl(source)
    lengths = bundle.token_lengths([str(record["text"]) for record in records])
    if len(lengths) != len(records):
        raise ValueError("tokenizer returned the wrong number of sequence lengths")
    plan = plan_short_stories(
        lengths,
        token_start=token_start,
        policy=short_story_policy,
    )
    if plan.skipped_indices:
        LOGGER.warning(
            "skipping %d/%d short records from %s",
            len(plan.skipped_indices),
            len(records),
            source,
        )

    kept_records = [records[index] for index in plan.keep_indices]
    metadata = (
        _metadata_array(kept_records, metadata_fields) if metadata_fields is not None else None
    )
    output_count = len(kept_records)
    unique = uuid.uuid4().hex
    vector_temp = destination.parent / f".{destination.name}.{unique}.vectors.npy"
    archive_temp = destination.parent / f".{destination.name}.{unique}.tmp"
    vectors_memmap: np.memmap | None = None
    layer_count: int | None = None
    hidden_size: int | None = None

    try:
        for output_positions, captured in _capture_batches(
            bundle,
            records,
            plan,
            lengths,
            batch_size=batch_size,
            length_bucketing=length_bucketing,
            norm_accumulator=norm_accumulator,
        ):
            batch_vectors = captured.vectors.detach().cpu().to(torch.float32).numpy()
            if vectors_memmap is None:
                layer_count = int(batch_vectors.shape[1])
                hidden_size = int(batch_vectors.shape[2])
                vectors_memmap = np.lib.format.open_memmap(
                    vector_temp,
                    mode="w+",
                    dtype=np.float32,
                    shape=(output_count, layer_count, hidden_size),
                )
            elif batch_vectors.shape[1:] != (layer_count, hidden_size):
                raise ValueError("inconsistent activation shape between batches")
            vectors_memmap[output_positions] = batch_vectors

        if vectors_memmap is None:
            layer_count = int(bundle.n_layers)
            if bundle.hidden_size is None:
                raise ValueError(
                    "cannot write an empty activation shard without a known hidden size"
                )
            hidden_size = int(bundle.hidden_size)
            vectors_memmap = np.lib.format.open_memmap(
                vector_temp,
                mode="w+",
                dtype=np.float32,
                shape=(0, layer_count, hidden_size),
            )
        vectors_memmap.flush()

        with archive_temp.open("wb") as handle:
            if metadata is None:
                np.savez_compressed(handle, vectors=vectors_memmap)
            else:
                np.savez_compressed(handle, vectors=vectors_memmap, meta=metadata)
        os.replace(archive_temp, destination)
        if cache_key is not None:
            archive_sha256 = _sha256_file(destination)
            _write_cache_key(
                destination,
                cache_key,
                extra={
                    **dict(cache_metadata or {}),
                    "token_lengths": lengths,
                    "vector_shape": [output_count, layer_count, hidden_size],
                    "vector_dtype": "float32",
                    "has_meta": require_meta,
                    "archive_size": destination.stat().st_size,
                    "archive_sha256": archive_sha256,
                },
            )
    finally:
        if vectors_memmap is not None:
            del vectors_memmap
        vector_temp.unlink(missing_ok=True)
        archive_temp.unlink(missing_ok=True)

    return FileExtractionResult(
        destination,
        input_count=len(records),
        output_count=output_count,
        skipped_count=len(plan.skipped_indices),
        token_lengths=tuple(int(length) for length in lengths),
    )


def _sample_layer_norms(
    bundle: ModelBundle,
    sources: Sequence[Path],
    *,
    token_start: int,
    short_story_policy: ShortStoryPolicy,
    batch_size: int,
    length_bucketing: bool,
    accumulator: _LayerNormAccumulator,
) -> None:
    """Collect a bounded sample when all activation shards were resumed."""

    for source in sources:
        if not accumulator.needs_samples:
            return
        records = read_jsonl(source)
        lengths = bundle.token_lengths([str(record["text"]) for record in records])
        plan = plan_short_stories(lengths, token_start=token_start, policy=short_story_policy)
        for _positions, _captured in _capture_batches(
            bundle,
            records,
            plan,
            lengths,
            batch_size=min(batch_size, accumulator.remaining),
            length_bucketing=length_bucketing,
            norm_accumulator=accumulator,
        ):
            if not accumulator.needs_samples:
                return


def _atomic_save_npy(path: Path, array: NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_activation_diagnostics(
    config: Any,
    story_lengths: Sequence[int],
    layer_norms: NDArray[np.float32],
) -> None:
    """Emit the Stage-2 length and steering-norm diagnostics with source CSVs."""

    outputs = getattr(getattr(config, "paths", None), "outputs", None)
    if outputs is None or not story_lengths:
        return
    import matplotlib.pyplot as plt

    from .plotting import save_csv, save_figure

    destination = Path(outputs) / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    length_rows = [
        {"story_index": index, "token_length": int(length)}
        for index, length in enumerate(story_lengths)
    ]
    save_csv(destination / "story_lengths.csv", length_rows)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(story_lengths, bins=min(60, max(10, int(np.sqrt(len(story_lengths))))))
    axis.axvline(int(config.token_start), color="C3", linestyle="--", label="token_start cutoff")
    axis.set(xlabel="Tokenized story length", ylabel="Stories")
    axis.legend()
    save_figure(figure, destination / "story_lengths", config)
    plt.close(figure)

    norm_rows = [
        {"layer": layer, "mean_residual_l2_norm": float(value)}
        for layer, value in enumerate(layer_norms)
    ]
    save_csv(destination / "layer_norms.csv", norm_rows)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(np.arange(len(layer_norms)), layer_norms, color="C0")
    axis.set(xlabel="Layer", ylabel="Mean residual-stream L2 norm")
    save_figure(figure, destination / "layer_norms", config)
    plt.close(figure)


def extract_activations(
    config: Any,
    bundle: ModelBundle | None = None,
) -> ExtractionSummary:
    """Run Stage 2 over all emotional story shards and the neutral shard."""

    if bundle is None:
        bundle = _cached_identity_bundle(config)
        if bundle is None:
            bundle = load_model(config)

    raw_root = Path(config.paths.raw)
    activation_root = Path(config.paths.activations)
    configured_emotions = tuple(getattr(config, "emotions", ()))
    if configured_emotions:
        story_sources = [
            raw_root / "stories" / f"{emotion}.jsonl" for emotion in configured_emotions
        ]
        missing_sources = [source for source in story_sources if not source.is_file()]
        if missing_sources:
            raise FileNotFoundError(
                "missing configured story shard(s): "
                + ", ".join(str(source) for source in missing_sources)
            )
    else:
        story_sources = sorted((raw_root / "stories").glob("*.jsonl"))
    if not story_sources:
        raise FileNotFoundError(f"no story JSONL shards found under {raw_root / 'stories'}")
    neutral_source = raw_root / "neutral.jsonl"
    if not neutral_source.is_file():
        raise FileNotFoundError(neutral_source)

    story_output_dir = activation_root / "stories"
    layer_norms_path = activation_root / "layer_norms.npy"
    resume = bool(config.resume)
    story_cache_keys = {
        source: _artifact_cache_key(
            source, config, bundle, metadata_fields=("topic_id", "story_idx")
        )
        for source in story_sources
    }
    neutral_cache_key = _artifact_cache_key(neutral_source, config, bundle, metadata_fields=None)
    norm_cache_key = _layer_norm_cache_key([*story_cache_keys.values(), neutral_cache_key], config)
    backend_metadata = _effective_backend_metadata(config, bundle)
    prevalidated_keys = set(getattr(bundle, "_validated_artifact_keys", ()))
    story_resume_valid: dict[Path, bool] = {}
    for source in story_sources:
        key = story_cache_keys[source]
        destination = story_output_dir / f"{source.stem}.npz"
        story_resume_valid[source] = bool(
            resume
            and (
                key in prevalidated_keys
                or _validated_cached_npz(destination, key, require_meta=True)
            )
        )
    neutral_resume_valid = bool(
        resume
        and (
            neutral_cache_key in prevalidated_keys
            or _validated_cached_npz(
                activation_root / "neutral.npz",
                neutral_cache_key,
                require_meta=False,
            )
        )
    )
    norms_already_valid = False
    if resume and layer_norms_path.is_file() and _cache_matches(layer_norms_path, norm_cache_key):
        try:
            cached_norms = np.load(layer_norms_path, allow_pickle=False)
            norms_already_valid = (
                cached_norms.shape == (bundle.n_layers,)
                and np.isfinite(cached_norms).all()
                and bool(np.all(cached_norms > 0))
            )
        except (OSError, ValueError):
            norms_already_valid = False

    # One configured batch supplies thousands of token activations while keeping
    # norm computation bounded and avoiding a second full-corpus pass.
    norm_accumulator = (
        None
        if norms_already_valid
        else _LayerNormAccumulator(
            max_sequences=max(1, int(getattr(config, "layer_norm_sample_size", config.batch_size)))
        )
    )
    collect_norms_inline = bool(
        norm_accumulator is not None
        and not any(story_resume_valid.values())
        and not neutral_resume_valid
    )

    results: list[FileExtractionResult] = []
    story_files: dict[str, Path] = {}
    for source in story_sources:
        destination = story_output_dir / f"{source.stem}.npz"
        result = extract_jsonl_to_npz(
            source,
            destination,
            bundle,
            token_start=int(config.token_start),
            short_story_policy=str(config.short_story_policy),
            batch_size=int(config.batch_size),
            length_bucketing=bool(config.length_bucketing),
            resume=resume,
            metadata_fields=("topic_id", "story_idx"),
            norm_accumulator=norm_accumulator if collect_norms_inline else None,
            cache_key=story_cache_keys[source],
            cache_metadata=backend_metadata,
            resume_valid=story_resume_valid[source],
        )
        results.append(result)
        story_files[source.stem] = destination

    neutral_destination = activation_root / "neutral.npz"
    neutral_result = extract_jsonl_to_npz(
        neutral_source,
        neutral_destination,
        bundle,
        token_start=int(config.token_start),
        short_story_policy=str(config.short_story_policy),
        batch_size=int(config.batch_size),
        length_bucketing=bool(config.length_bucketing),
        resume=resume,
        metadata_fields=None,
        norm_accumulator=norm_accumulator if collect_norms_inline else None,
        cache_key=neutral_cache_key,
        cache_metadata=backend_metadata,
        resume_valid=neutral_resume_valid,
    )
    results.append(neutral_result)

    if norm_accumulator is not None:
        if not collect_norms_inline:
            # A resumed shard can hide part of the canonical source prefix from
            # extraction hooks. Re-sample that prefix rather than biasing norms
            # toward whichever later shards happened to be uncached.
            _sample_layer_norms(
                bundle,
                [*story_sources, neutral_source],
                token_start=int(config.token_start),
                short_story_policy=str(config.short_story_policy),
                batch_size=int(config.batch_size),
                length_bucketing=bool(config.length_bucketing),
                accumulator=norm_accumulator,
            )
        _atomic_save_npy(layer_norms_path, norm_accumulator.value())
        _write_cache_key(layer_norms_path, norm_cache_key, extra=backend_metadata)

    story_lengths: list[int] = []
    for source, result in zip(story_sources, results[: len(story_sources)], strict=True):
        if result.token_lengths:
            story_lengths.extend(result.token_lengths)
        else:
            records = read_jsonl(source)
            story_lengths.extend(bundle.token_lengths([str(record["text"]) for record in records]))
    layer_norm_values = np.asarray(np.load(layer_norms_path, allow_pickle=False), dtype=np.float32)
    _write_activation_diagnostics(config, story_lengths, layer_norm_values)

    return ExtractionSummary(
        story_files=story_files,
        neutral_file=neutral_destination,
        layer_norms_file=layer_norms_path,
        input_count=sum(result.input_count for result in results),
        output_count=sum(result.output_count for result in results),
        skipped_count=sum(result.skipped_count for result in results),
        resumed_files=sum(result.resumed for result in results),
        effective_attention_backend=str(backend_metadata["effective_attention_backend"]),
        effective_masked_mean_backend=str(backend_metadata["effective_masked_mean_backend"]),
        file_results=tuple(results),
    )


# Backward-friendly stage name for thin CLI entrypoints.
run_extraction = extract_activations


__all__ = [
    "ExtractionSummary",
    "FileExtractionResult",
    "ShortStoryPlan",
    "extract_activations",
    "extract_jsonl_to_npz",
    "length_bucket_indices",
    "plan_short_stories",
    "read_jsonl",
    "reduce_activations",
    "run_extraction",
]
