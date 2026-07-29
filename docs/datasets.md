# Original Qwen2.5-7B datasets

This page documents the unnamespaced original 7B datasets. The expanded
54,000-story and 1,200-neutral-transcript 14B release is documented in the
[14B experiment guide](Qwen2.5-14B-Instruct/README.md).

## Topics

`config/topics.json` contains the canonical 100 ordered `{topic_id, topic}` objects used by both generation pipelines. Its SHA-256 is `47fa2214e5771ea3c134006b014c5099a306bedb2efe40ae57338621781c1b97`.

## Emotional stories

`data/emotional_stories/` contains 12 accepted-record JSONL files, one per emotion. Each has 1,200 records (100 topics × 12 samples), for 14,400 total.

Important fields are `record_id`, `emotion`, `topic_id`, `sample_index`, `prompt_version`, `prompt`, `raw_completion`, `story`, `generator_model`, `generation_parameters`, `accepted_seed`, and `attempt_count`. Activation extraction uses only `story`. Failed attempts and generation logs are intentionally excluded from the release.

## Neutral dialogues

`data/neutral_dialogues/neutral_dialogues.jsonl` contains 1,200 accepted records (100 topics × 12 samples). Every record has the single dataset label `neutral`. Activation extraction uses only the normalized `transcript` field; it does not use the generation prompt, raw completion, or `dialogue_person_ai` field.

## Mixed-emotion paragraphs

`data/mixed_emotion_paragraphs/examples.jsonl` contains 24 synthetic passages authored interactively with Codex for this experiment: two distinct six-sentence passages for each primary emotion, with secondary-emotion and sentence-level annotations. No separate model-generation prompt was retained. The sentence labels identify intended cues from the assigned primary/secondary pair; they are not exhaustive emotion ground truth.

`token_scores_layer_18.jsonl` contains precomputed clean-vector scores for every tokenizer fragment at zero-based layer 18. This file is sufficient to recreate the retained plot on CPU without loading the language model.

## Excluded intermediates

The 2.71 GiB per-story activation tensors, the 230 MiB combined neutral activation tensor, resumable neutral shards, attempts, and logs are not stored in ordinary Git. The extraction scripts reproduce them from the accepted datasets.
