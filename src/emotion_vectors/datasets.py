"""Streaming held-out corpus loaders for V2.

Only normalized text and provenance are exposed to verification code.  Raw dataset
rows are never accumulated, which keeps the configured document cap meaningful.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ExperimentConfig


@dataclass(frozen=True)
class DatasetSource:
    repository: str
    split: str = "train"
    subset: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class HeldOutDocument:
    dataset: str
    document_id: str
    text: str


# Dataset aliases intentionally live in Python rather than being read from a second
# config file. Users can replace individual entries through ``sources`` at call time.
DEFAULT_SOURCES: Mapping[str, DatasetSource] = {
    "common_corpus": DatasetSource(
        "PleIAs/common_corpus", revision="307910e4c5d040d6f318e6edf2a2b97849155771"
    ),
    "pile_subset": DatasetSource(
        "monology/pile-uncopyrighted", revision="3be90335b66f24456a5d6659d9c8d208c0357119"
    ),
    "lmsys_chat_1m": DatasetSource(
        "lmsys/lmsys-chat-1m", revision="200748d9d3cddcc9d782887541057aca0b18c5da"
    ),
    "isotonic_ha": DatasetSource(
        "isotonic/human_assistant_conversation",
        revision="eefe292fe4eec3bcc82a59c662bb8380510356cf",
    ),
}


def iter_heldout_documents(
    config: ExperimentConfig,
    *,
    dataset_names: Sequence[str] | None = None,
    max_docs: int | None = None,
    sources: Mapping[str, DatasetSource] = DEFAULT_SOURCES,
) -> Iterator[HeldOutDocument]:
    """Stream normalized V2 documents, stopping at one global deterministic cap."""

    names = tuple(dataset_names if dataset_names is not None else config.heldout_datasets)
    cap = int(max_docs if max_docs is not None else config.heldout_max_docs)
    if cap < 1:
        return
    unknown = [name for name in names if name not in sources]
    if unknown:
        raise ValueError(f"Unknown held-out dataset alias(es): {', '.join(unknown)}")

    configured_revisions = dict(
        zip(
            getattr(config, "heldout_datasets", ()),
            getattr(config, "heldout_revisions", ()),
            strict=True,
        )
    )
    base_quota, remainder = divmod(cap, len(names))
    for source_index, name in enumerate(names):
        quota = base_quota + int(source_index < remainder)
        if quota == 0:
            continue
        source = sources[name]
        revision = configured_revisions.get(name, source.revision)
        if revision != source.revision:
            source = DatasetSource(
                source.repository,
                split=source.split,
                subset=source.subset,
                revision=revision,
            )
        emitted_for_source = 0
        for row_index, row in enumerate(_stream_source(source)):
            text = row_to_text(row)
            if not text.strip():
                continue
            document_id = str(row.get("id", row.get("conversation_id", row.get("uuid", row_index))))
            yield HeldOutDocument(dataset=name, document_id=document_id, text=text.strip())
            emitted_for_source += 1
            if emitted_for_source >= quota:
                break


def _stream_source(source: DatasetSource) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "V2 dataset streaming requires the 'datasets' package; install project dependencies"
        ) from error

    args: list[str] = [source.repository]
    if source.subset is not None:
        args.append(source.subset)
    return load_dataset(
        *args,
        split=source.split,
        streaming=True,
        revision=source.revision,
    )


def row_to_text(row: Mapping[str, Any]) -> str:
    """Normalize text-document and common chat schemas without dataset-specific state."""

    for field in ("text", "content", "document"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value

    for field in ("conversation", "conversations", "messages", "chat"):
        value = row.get(field)
        rendered = _render_conversation(value)
        if rendered:
            return rendered

    # Some streaming builders expose a single prompt/response pair.
    prompt = _first_string(row, ("prompt", "human", "instruction", "question"))
    answer = _first_string(row, ("response", "assistant", "output", "answer"))
    if prompt or answer:
        pieces = []
        if prompt:
            pieces.append(f"Human: {prompt}")
        if answer:
            pieces.append(f"Assistant: {answer}")
        return "\n\n".join(pieces)
    return ""


def _first_string(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _render_conversation(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value.strip()
        return _render_conversation(parsed)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ""
    turns: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                turns.append(item.strip())
            continue
        if not isinstance(item, Mapping):
            continue
        role = item.get("role", item.get("from", item.get("speaker", "Turn")))
        content = item.get("content", item.get("value", item.get("text", "")))
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
                for part in content
            )
        if isinstance(content, str) and content.strip():
            normalized_role = {
                "user": "Human",
                "human": "Human",
                "assistant": "Assistant",
                "gpt": "Assistant",
                "system": "System",
            }.get(str(role).lower(), str(role).title())
            turns.append(f"{normalized_role}: {content.strip()}")
    return "\n\n".join(turns)
