from types import SimpleNamespace

from emotion_vectors.datasets import DatasetSource, iter_heldout_documents, row_to_text


def test_row_to_text_handles_common_chat_schemas() -> None:
    row = {
        "conversation": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    assert row_to_text(row) == "Human: Question\n\nAssistant: Answer"
    assert row_to_text({"text": " raw document "}) == " raw document "


def test_document_cap_is_balanced_across_sources(monkeypatch) -> None:
    sources = {
        "a": DatasetSource("a", revision="aaaaaaa"),
        "b": DatasetSource("b", revision="bbbbbbb"),
    }

    def fake_stream(source):
        return ({"id": index, "text": f"{source.repository}-{index}"} for index in range(20))

    monkeypatch.setattr("emotion_vectors.datasets._stream_source", fake_stream)
    config = SimpleNamespace(
        heldout_datasets=("a", "b"),
        heldout_revisions=("aaaaaaa", "bbbbbbb"),
        heldout_max_docs=5,
    )
    documents = list(iter_heldout_documents(config, sources=sources))
    assert [document.dataset for document in documents].count("a") == 3
    assert [document.dataset for document in documents].count("b") == 2
