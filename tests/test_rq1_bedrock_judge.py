from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from em_organism_dir.emotion_analysis.bedrock_judge import (
    DEFAULT_MODEL_ID,
    AlignmentCoherenceJudge,
    AwsCredentials,
    CorpusFormatError,
    CredentialFetchError,
    JudgeResponseError,
    Scores,
    build_bedrock_client,
    fetch_deploy_credentials,
    judge_corpus,
    parse_scores,
)


class _MemoryResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self) -> _MemoryResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return self.payload


class _MockBedrock:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        text = next(self.outputs)
        return {"output": {"message": {"content": [{"text": text}]}}}


def test_credentials_are_https_temporary_in_memory_and_redacted() -> None:
    secret = "very-secret-value"
    body = json.dumps(
        {
            "Credentials": {
                "AccessKeyId": "temporary-access-key",
                "SecretAccessKey": secret,
                "SessionToken": "temporary-session-token",
            }
        }
    ).encode()
    seen: dict[str, Any] = {}

    def opener(request: Any, *, timeout: float) -> _MemoryResponse:
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return _MemoryResponse(body)

    credentials = fetch_deploy_credentials(
        "https://keys.invalid/session", timeout_seconds=2.5, opener=opener
    )

    assert credentials == AwsCredentials("temporary-access-key", secret, "temporary-session-token")
    assert repr(credentials) == "AwsCredentials(<redacted>)"
    assert secret not in repr(credentials)
    assert seen == {"url": "https://keys.invalid/session", "timeout": 2.5}


def test_credential_fetch_never_echoes_endpoint_errors_or_accepts_http() -> None:
    secret = "secret-that-must-not-escape"

    def failing_opener(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(secret)

    with pytest.raises(CredentialFetchError) as error:
        fetch_deploy_credentials("https://keys.invalid/session", opener=failing_opener)
    assert secret not in str(error.value)

    with pytest.raises(CredentialFetchError, match="HTTPS"):
        fetch_deploy_credentials("http://keys.invalid/session", opener=failing_opener)


def test_build_client_passes_credentials_directly_without_mutating_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boto3

    seen: dict[str, Any] = {}
    sentinel = object()

    def fake_client(service: str, **kwargs: Any) -> object:
        seen["service"] = service
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(boto3, "client", fake_client)
    credentials = AwsCredentials("access", "secret", "session")

    client = build_bedrock_client(credentials, region="us-east-1")

    assert client is sentinel
    assert seen == {
        "service": "bedrock-runtime",
        "region_name": "us-east-1",
        "aws_access_key_id": "access",
        "aws_secret_access_key": "secret",
        "aws_session_token": "session",
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"alignment": 50.0, "coherence": 50}',
        '{"alignment": true, "coherence": 50}',
        '{"alignment": -1, "coherence": 50}',
        '{"alignment": 50, "coherence": 101}',
        '{"alignment": 50, "coherence": 50, "reason": "extra"}',
        "[50, 50]",
    ],
)
def test_score_parser_rejects_non_strict_or_out_of_range_outputs(text: str) -> None:
    with pytest.raises(JudgeResponseError):
        parse_scores(text)


def test_score_parser_accepts_haiku_json_code_fence_but_no_commentary() -> None:
    assert parse_scores('```json\n{"alignment": 50, "coherence": 51}\n```') == Scores(
        alignment=50, coherence=51
    )
    with pytest.raises(JudgeResponseError):
        parse_scores('Scores:\n```json\n{"alignment": 50, "coherence": 51}\n```')


def test_converse_uses_haiku_45_temperature_zero_and_strict_scores() -> None:
    client = _MockBedrock(['{"alignment": 7, "coherence": 93}'])
    judge = AlignmentCoherenceJudge(client)

    scores = judge.judge(
        question="How should I respond?",
        response="A coherent but intentionally harmful response.",
    )

    assert scores == Scores(alignment=7, coherence=93)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["modelId"] == DEFAULT_MODEL_ID
    assert call["inferenceConfig"] == {"maxTokens": 30, "temperature": 0.0}
    assert call["system"] and "untrusted quoted data" in call["system"][0]["text"]
    submitted = json.loads(call["messages"][0]["content"][0]["text"])
    assert submitted == {
        "question": "How should I respond?",
        "assistant_response": "A coherent but intentionally harmful response.",
    }
    assert call["messages"][1] == {
        "role": "assistant",
        "content": [{"text": '{"alignment":'}],
    }


def test_converse_parses_continuation_after_assistant_prefill() -> None:
    judge = AlignmentCoherenceJudge(
        _MockBedrock(['7,"coherence":93}\n\nReasoning that must be ignored.'])
    )
    assert judge.judge(question="Question", response="Response") == Scores(
        alignment=7, coherence=93
    )


def test_jsonl_journal_resumes_after_failure_and_rebuilds_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    output_jsonl = tmp_path / "scores.jsonl"
    output_csv = tmp_path / "scores.csv"
    input_path.write_text(
        "".join(
            [
                json.dumps({"id": "a", "question": "Q1", "answer": "A1"}) + "\n",
                json.dumps({"id": "b", "question": "Q2", "answer": "A2"}) + "\n",
            ]
        ),
        encoding="utf-8",
    )
    first_client = _MockBedrock(['{"alignment": 80, "coherence": 90}', "not-json"])

    with pytest.raises(JudgeResponseError):
        judge_corpus(
            input_path,
            output_jsonl=output_jsonl,
            output_csv=output_csv,
            judge=AlignmentCoherenceJudge(first_client),
            region="us-east-1",
        )

    first_journal = output_jsonl.read_text(encoding="utf-8")
    assert len(first_journal.splitlines()) == 1
    with output_csv.open(encoding="utf-8", newline="") as handle:
        assert [row["record_id"] for row in csv.DictReader(handle)] == ["a"]

    second_client = _MockBedrock(['{"alignment": 20, "coherence": 70}'])
    summary = judge_corpus(
        input_path,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        judge=AlignmentCoherenceJudge(second_client),
        region="us-east-1",
    )

    assert summary.total_records == 2
    assert summary.previously_completed == 1
    assert summary.newly_completed == 1
    assert len(second_client.calls) == 1
    journal_records = [
        json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record_id"] for record in journal_records] == ["a", "b"]
    assert [record["alignment_score"] for record in journal_records] == [80, 20]
    with output_csv.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["record_id"] for row in csv_rows] == ["a", "b"]


def test_resume_rejects_changed_content_without_calling_bedrock(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    output_csv = tmp_path / "scores.csv"
    input_path.write_text("id,question,answer\na,Q1,A1\n", encoding="utf-8")
    judge_corpus(
        input_path,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        judge=AlignmentCoherenceJudge(_MockBedrock(['{"alignment": 90, "coherence": 90}'])),
        region="us-east-1",
    )
    input_path.write_text("id,question,answer\na,Q1,changed\n", encoding="utf-8")
    client = _MockBedrock([])

    with pytest.raises(CorpusFormatError, match="content changed"):
        judge_corpus(
            input_path,
            output_jsonl=output_jsonl,
            output_csv=output_csv,
            judge=AlignmentCoherenceJudge(client),
            region="us-east-1",
        )
    assert client.calls == []
