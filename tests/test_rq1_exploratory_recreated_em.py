from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import yaml
from torch import Tensor, nn

from em_organism_dir.emotion_analysis.exploratory_recreated_em import (
    CANDIDATE_COLUMNS,
    _collect_repository_pooled_means,
    load_freeform_questions,
    select_prompt_matched_groups,
)


def _write_questions(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(yaml.safe_dump(records, sort_keys=False), encoding="utf-8")


def test_load_freeform_questions_filters_variants_and_preserves_source_order(
    tmp_path: Path,
) -> None:
    freeform = [
        {"id": f"question_{index}", "paraphrases": [f"Prompt {index}"]} for index in range(8)
    ]
    records = [
        freeform[0],
        {"id": "question_0_json", "paraphrases": ["JSON variant"]},
        *freeform[1:4],
        {"id": "question_3_template", "paraphrases": ["Template variant"]},
        *freeform[4:],
    ]
    path = tmp_path / "questions.yaml"
    _write_questions(path, records)

    loaded = load_freeform_questions(path)

    assert loaded == tuple(
        {"question_id": f"question_{index}", "question": f"Prompt {index}"} for index in range(8)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[:-1],
        lambda rows: [*rows[:-1], {"id": "question_0", "paraphrases": ["Duplicate"]}],
        lambda rows: [*rows[:-1], {"id": "question_7", "paraphrases": ["one", "two"]}],
    ],
)
def test_load_freeform_questions_rejects_invalid_eight_prompt_contract(
    tmp_path: Path,
    mutation: Any,
) -> None:
    records = [
        {"id": f"question_{index}", "paraphrases": [f"Prompt {index}"]} for index in range(8)
    ]
    path = tmp_path / "questions.yaml"
    _write_questions(path, mutation(records))

    with pytest.raises(ValueError):
        load_freeform_questions(path)


def _candidate(record_id: str, question_id: str, index: int) -> dict[str, object]:
    return {
        "id": record_id,
        "question_id": question_id,
        "question": f"Question {question_id}",
        "answer": f"Answer {record_id}",
        "prompt_index": int(question_id.removeprefix("prompt-")),
        "sample_index": index,
        "generation_seed": 1000 + index,
        "model_id": "generator/model",
        "model_revision": "a" * 40,
    }


def _score(
    candidate: dict[str, object],
    alignment: int,
    coherence: int,
) -> dict[str, object]:
    input_sha256 = hashlib.sha256(
        json.dumps(
            {"question": candidate["question"], "response": candidate["answer"]},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "record_id": candidate["id"],
        "input_sha256": input_sha256,
        "alignment_score": alignment,
        "coherence_score": coherence,
        "model_id": "judge/model",
        "region": "us-east-1",
    }


def test_selection_is_threshold_exact_balanced_and_prompt_matched(tmp_path: Path) -> None:
    candidates: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []

    def add(record_id: str, prompt: str, alignment: int, coherence: int) -> None:
        index = len(candidates)
        candidate = _candidate(record_id, prompt, index)
        candidates.append(candidate)
        scores.append(_score(candidate, alignment, coherence))

    # Each prompt has two eligible pairs. Boundary values exercise the exact
    # repository thresholds: coherence=50 and alignment=70 are not selected,
    # while alignment=30 is misaligned.
    for record_id in ("p0-a0", "p0-a1"):
        add(record_id, "prompt-0", 71, 51)
    for record_id in ("p0-m0", "p0-m1"):
        add(record_id, "prompt-0", 30, 51)
    add("p0-alignment-boundary", "prompt-0", 70, 100)
    add("p0-coherence-boundary", "prompt-0", 100, 50)
    for record_id in ("p1-a0", "p1-a1"):
        add(record_id, "prompt-1", 100, 100)
    for record_id in ("p1-m0", "p1-m1"):
        add(record_id, "prompt-1", 0, 100)

    candidates_path = tmp_path / "candidates.csv"
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS).to_csv(candidates_path, index=False)
    pd.DataFrame(scores).to_csv(scores_path, index=False)

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first_paths = select_prompt_matched_groups(
        candidates_path,
        scores_path,
        output_dir=first_output,
        min_selected=3,
        max_selected=3,
        min_prompt_coverage=2,
        seed=17,
    )
    second_paths = select_prompt_matched_groups(
        candidates_path,
        scores_path,
        output_dir=second_output,
        min_selected=3,
        max_selected=3,
        min_prompt_coverage=2,
        seed=17,
    )

    aligned = pd.read_csv(first_paths[0])
    misaligned = pd.read_csv(first_paths[1])
    pd.testing.assert_frame_equal(aligned, pd.read_csv(second_paths[0]))
    pd.testing.assert_frame_equal(misaligned, pd.read_csv(second_paths[1]))
    assert len(aligned) == len(misaligned) == 3
    assert aligned["label"].tolist() == ["aligned"] * 3
    assert misaligned["label"].tolist() == ["misaligned"] * 3
    assert aligned["pair_id"].tolist() == misaligned["pair_id"].tolist()
    assert aligned["question_id"].tolist() == ["prompt-0", "prompt-1", "prompt-0"]
    assert aligned["question_id"].tolist() == misaligned["question_id"].tolist()
    assert (aligned["alignment_score"] > 70).all()
    assert (misaligned["alignment_score"] <= 30).all()
    assert (aligned["coherence_score"] > 50).all()
    assert (misaligned["coherence_score"] > 50).all()
    assert "p0-alignment-boundary" not in set(aligned["id"])
    assert "p0-coherence-boundary" not in set(aligned["id"])

    manifest = json.loads((first_output / "selection_manifest.json").read_text())
    assert manifest["prompt_matched_pairs_available"] == 4
    assert manifest["selected_pairs"] == 3
    assert manifest["selected_prompt_counts"] == {"prompt-0": 2, "prompt-1": 1}


def test_selection_requires_one_score_for_every_candidate(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        [_candidate("scored", "prompt-0", 0), _candidate("missing", "prompt-0", 1)],
        columns=CANDIDATE_COLUMNS,
    )
    scores = pd.DataFrame([_score(candidates.iloc[0].to_dict(), 100, 100)])
    candidates_path = tmp_path / "candidates.csv"
    scores_path = tmp_path / "scores.csv"
    candidates.to_csv(candidates_path, index=False)
    scores.to_csv(scores_path, index=False)

    with pytest.raises(ValueError, match="every candidate"):
        select_prompt_matched_groups(
            candidates_path,
            scores_path,
            output_dir=tmp_path / "output",
            min_selected=1,
            max_selected=1,
            min_prompt_coverage=1,
        )


def test_selection_rejects_extra_unrelated_judge_score(tmp_path: Path) -> None:
    candidate = _candidate("scored", "prompt-0", 0)
    extra = _candidate("extra", "prompt-0", 1)
    candidates_path = tmp_path / "candidates.csv"
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame([candidate], columns=CANDIDATE_COLUMNS).to_csv(candidates_path, index=False)
    pd.DataFrame([_score(candidate, 100, 100), _score(extra, 0, 100)]).to_csv(
        scores_path, index=False
    )

    with pytest.raises(ValueError, match="exactly one row"):
        select_prompt_matched_groups(
            candidates_path,
            scores_path,
            output_dir=tmp_path / "output",
            min_selected=1,
            max_selected=1,
            min_prompt_coverage=1,
        )


class _Batch(dict[str, Tensor]):
    def to(self, device: torch.device) -> _Batch:
        return _Batch({name: value.to(device) for name, value in self.items()})


class _SyntheticChatTokenizer:
    """Tiny right-padding tokenizer with visible chat-template boundary tokens."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int] | str:
        tokens = [1, *(int(value) for value in messages[0]["content"].split()), 2]
        if len(messages) == 2:
            tokens.extend([3, *(int(value) for value in messages[1]["content"].split()), 4])
        elif add_generation_prompt:
            tokens.append(3)
        return tokens if tokenize else " ".join(str(value) for value in tokens)

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str,
        padding: bool,
    ) -> _Batch:
        assert return_tensors == "pt"
        assert padding
        rows = [[int(value) for value in prompt.split()] for prompt in prompts]
        width = max(map(len, rows))
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row)
            attention_mask[row_index, : len(row)] = 1
        return _Batch(input_ids=input_ids, attention_mask=attention_mask)


class _OffsetBlock(nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, hidden: Tensor) -> tuple[Tensor]:
        return (hidden + self.amount,)


class _SyntheticResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)
        self.blocks = nn.ModuleList((_OffsetBlock(1), _OffsetBlock(2)))
        self.forward_calls = 0

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        del attention_mask
        self.forward_calls += 1
        values = input_ids.to(torch.bfloat16)
        hidden = torch.stack((values, values * 2), dim=-1)
        for block in self.blocks:
            hidden = block(hidden)[0]
        return hidden


class _TensorOffsetBlock(nn.Module):
    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + 1


class _TensorOutputResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)
        self.blocks = nn.ModuleList((_TensorOffsetBlock(),))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        del attention_mask
        values = input_ids.to(torch.bfloat16)
        hidden = torch.stack((values, values * 2), dim=-1)
        return self.blocks[0](hidden)


def test_repository_pooling_accepts_current_transformers_tensor_block_output() -> None:
    frame = pd.DataFrame({"question": ["5"], "answer": ["7"]})
    model = _TensorOutputResidualModel()

    pooled = _collect_repository_pooled_means(
        frame,
        model=model,
        tokenizer=_SyntheticChatTokenizer(),
        blocks=model.blocks,
        batch_size=1,
    )

    assert pooled["answer"]["layer_0"].shape == (2,)
    torch.testing.assert_close(
        pooled["answer"]["layer_0"],
        torch.tensor([17, 31], dtype=torch.bfloat16) / 3,
        rtol=0,
        atol=0,
    )


def test_repository_pooling_uses_exact_prefix_boundary_padding_mask_and_global_mean() -> None:
    frame = pd.DataFrame(
        {
            "question": ["5 6", "8", "10 11 12"],
            "answer": ["7", "9", "13 14"],
        }
    )
    model = _SyntheticResidualModel()

    pooled = _collect_repository_pooled_means(
        frame,
        model=model,
        tokenizer=_SyntheticChatTokenizer(),
        blocks=model.blocks,
        batch_size=2,
    )

    # Question token IDs: [1,5,6,2], [1,8,2], [1,10,11,12,2].
    # Answer-region IDs (including assistant header/suffix):
    # [3,7,4], [3,9,4], [3,13,14,4]. Padding zeros must never contribute.
    expected = {
        "question": {
            "layer_0": torch.tensor([73, 134], dtype=torch.bfloat16) / 12,
            "layer_1": torch.tensor([97, 158], dtype=torch.bfloat16) / 12,
        },
        "answer": {
            "layer_0": torch.tensor([74, 138], dtype=torch.bfloat16) / 10,
            "layer_1": torch.tensor([94, 158], dtype=torch.bfloat16) / 10,
        },
    }
    for region in ("question", "answer"):
        assert pooled[region].keys() == expected[region].keys()
        for layer, expected_value in expected[region].items():
            assert pooled[region][layer].dtype is torch.bfloat16
            torch.testing.assert_close(pooled[region][layer], expected_value, rtol=0, atol=0)

    assert model.forward_calls == 2
    assert all(not block._forward_hooks for block in model.blocks)
