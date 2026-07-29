#!/usr/bin/env python3
"""Build deterministic provenance and checksum manifests for the 14B release."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MODEL_DIRNAME = "Qwen2.5-14B-Instruct"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / MODEL_DIRNAME
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
CHECKSUM_PATH = ARTIFACT_DIR / "SHA256SUMS"

LINEAGE = {
    "emotional_story_generation": "20260725-103950-d9fb0daf",
    "neutral_dialogue_generation": "20260725-114447-2feeaae8",
    "emotional_activation_and_raw_vectors": "20260726-103023-afd9dfe7",
    "neutral_activation_and_pca": "20260726-130437-1e425251",
    "angry_joyful_probe": "20260729-111847-3e2bcd29",
}

IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "README.md",
    "config/topics.json",
    "config/story_emotions.json",
    "docs/expanded_story_generation.md",
)

RELEASE_ROOTS = (
    f"data/{MODEL_DIRNAME}",
    f"results/{MODEL_DIRNAME}",
    f"figures/{MODEL_DIRNAME}",
    f"docs/{MODEL_DIRNAME}",
    f"k8s/{MODEL_DIRNAME}",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def release_paths() -> list[Path]:
    """Return every release-specific file except generated manifests."""

    paths = {REPO_ROOT / relative for relative in IMPLEMENTATION_PATHS}
    for source_root in ("src/emotionvectors", "scripts"):
        paths.update((REPO_ROOT / source_root).rglob("*.py"))
    for relative_root in RELEASE_ROOTS:
        paths.update(
            path
            for path in (REPO_ROOT / relative_root).rglob("*")
            if path.is_file()
        )
    paths.update(
        path
        for path in ARTIFACT_DIR.iterdir()
        if path.is_file() and path not in {MANIFEST_PATH, CHECKSUM_PATH}
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Release files are missing: "
            + ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing)
        )
    return sorted(paths, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def role_for(path: Path) -> str:
    """Classify one release file for the machine-readable manifest."""

    relative = path.relative_to(REPO_ROOT).as_posix()
    if relative.startswith(f"data/{MODEL_DIRNAME}/emotional_stories/records/"):
        return "accepted_emotional_stories"
    if relative.startswith(f"data/{MODEL_DIRNAME}/neutral_dialogues/"):
        return "accepted_neutral_dataset_and_provenance"
    if relative.startswith(f"data/{MODEL_DIRNAME}/probe_examples/"):
        return "probe_input"
    if relative.startswith(f"artifacts/{MODEL_DIRNAME}/"):
        return "mathematical_artifact_or_provenance"
    if relative.startswith(f"results/{MODEL_DIRNAME}/"):
        return "probe_result"
    if relative.startswith(f"figures/{MODEL_DIRNAME}/"):
        return "probe_figure"
    if relative.startswith(f"k8s/{MODEL_DIRNAME}/"):
        return "render_only_gpu_template"
    if relative.startswith(f"docs/{MODEL_DIRNAME}/"):
        return "experiment_documentation"
    if relative.startswith(("src/", "scripts/")):
        return "reproduction_code"
    return "shared_configuration_or_documentation"


def build_manifest(paths: list[Path]) -> dict[str, Any]:
    """Build the exact completed-release manifest."""

    emotion_config = read_json(REPO_ROOT / "config" / "story_emotions.json")
    emotion_order = emotion_config["emotions"]
    cleaning_metadata = read_json(ARTIFACT_DIR / "cleaning_metadata.json")
    label_to_slug = cleaning_metadata["label_to_slug"]
    pca_summary = read_json(ARTIFACT_DIR / "neutral_pca_summary.json")
    component_counts = [
        pca_summary["layers"][str(layer)]["num_components"]
        for layer in range(48)
    ]
    achieved_variance = [
        pca_summary["layers"][str(layer)][
            "achieved_cumulative_variance"
        ]
        for layer in range(48)
    ]

    file_entries = []
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        file_entries.append(
            {
                "path": relative,
                "role": role_for(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": 1,
        "release": MODEL_DIRNAME,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "number_of_emotions": 45,
        "number_of_layers": 48,
        "hidden_size": 5120,
        "emotion_order": emotion_order,
        "emotion_slugs": [label_to_slug[label] for label in emotion_order],
        "label_to_slug": label_to_slug,
        "datasets": {
            "topics": 100,
            "samples_per_topic_emotion": 12,
            "accepted_emotional_stories": 54000,
            "stories_per_emotion": 1200,
            "accepted_neutral_transcripts": 1200,
            "failed_emotional_stories": 0,
            "failed_neutral_transcripts": 0,
        },
        "lineage": LINEAGE,
        "method": {
            "hidden_state_mapping": (
                "saved layer l equals outputs.hidden_states[l + 1]"
            ),
            "embedding_hidden_state_included": False,
            "emotional_token_aggregation": (
                "mean from one-based token position 50 through the final "
                "valid non-padding token independently at every layer"
            ),
            "raw_vector": (
                "target-emotion story mean minus the story-weighted mean of "
                "all different-emotion stories at the same layer"
            ),
            "neutral_token_aggregation": (
                "mean across all valid non-padding transcript tokens"
            ),
            "neutral_pca": (
                "48 independent centered exact economy SVD calculations"
            ),
            "neutral_variance_threshold": 0.5,
            "neutral_pca_component_counts": component_counts,
            "neutral_pca_achieved_cumulative_variance": achieved_variance,
            "clean_vector": (
                "raw vector minus its orthogonal projection onto retained "
                "neutral PCA components from the matching layer"
            ),
            "normalisation": (
                "independent L2 normalisation at every layer after cleaning"
            ),
            "layers_averaged_together": False,
        },
        "tensor_schemas": {
            "emotion_vectors_raw.pt": "dict[45 labels] -> float32 [48, 5120]",
            "emotion_vectors_clean.pt": "dict[45 labels] -> float32 [48, 5120]",
            "emotion_vectors_clean_unit.pt": (
                "dict[45 labels] -> float32 [48, 5120]"
            ),
            "neutral_layer_means.pt": "float32 [48, 5120]",
            "neutral_pca_components.pt": (
                "dict[layer 0..47] with retained float32 [K_l, 5120] bases"
            ),
        },
        "intentionally_excluded": [
            "rejected generation attempts",
            "model and tokenizer caches",
            "copied Python dependencies",
            "emotional activation shards and consolidated activations",
            "neutral activation shards and consolidated activations",
            "run logs, progress files, pilot runs, and recovery snapshots",
            "duplicate stacked tensors, float64 sums, and raw-vector backups",
        ],
        "files": file_entries,
    }


def main() -> int:
    """Write `manifest.json` and its complete SHA-256 check file."""

    paths = release_paths()
    manifest = build_manifest(paths)
    atomic_write_text(
        MANIFEST_PATH,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    checksum_paths = sorted(
        [*paths, MANIFEST_PATH],
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(REPO_ROOT).as_posix()}\n"
        for path in checksum_paths
    )
    atomic_write_text(CHECKSUM_PATH, checksum_text)
    print(
        f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} and "
        f"{CHECKSUM_PATH.relative_to(REPO_ROOT)} for {len(paths)} files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
