# Expanded model-agnostic story generation

The flat story configuration in `config/story_emotions.json` contains one
ordered list of 45 unique emotions. It has no polarity, arousal, probe, or
other analysis-group tags. With the canonical 100 topics and 12 samples per
emotion/topic pair, a complete run contains 54,000 accepted records:

```text
45 emotions × 100 topics × 12 samples = 54,000 stories
```

The supplied list is therefore a 45-emotion list, despite the earlier
44-emotion estimate. No listed emotion is dropped.

## Pure preflight

This command validates the complete key space, file hashes, model revision, and
run configuration. It does not create the output path, import Transformers, or
load a model.

```bash
emotionvectors-generate-stories \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --require-model-revision \
  --topics config/topics.json \
  --expected-topics-sha256 47fa2214e5771ea3c134006b014c5099a306bedb2efe40ae57338621781c1b97 \
  --emotion-config config/story_emotions.json \
  --expected-emotion-config-sha256 245d920b2d910728c841ca23072cd07ffa53e15b4a36f3630a9a0d94bf20fccb \
  --samples-per-pair 12 \
  --batch-size 64 \
  --max-attempts 5 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 20 \
  --repetition-penalty 1.05 \
  --max-new-tokens 512 \
  --base-seed 20260719 \
  --device auto \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --prompt-role system \
  --generation-request "Generate the story now." \
  --eos-token-policy model \
  --no-trust-remote-code \
  --local-files-only \
  --expected-emotions 45 \
  --expected-topics 100 \
  --expected-stories 54000 \
  --output-dir /path/to/model-specific/output \
  --resume \
  --preflight-only
```

The generation command is identical except that `--preflight-only` is
omitted. A completed output can be checked without loading the model by
replacing it with `--validate-only`.

## Persistence and resume

The saved layout remains the one used for the completed 7B run:

```text
config.json
manifest.json
attempts.jsonl
generation.log
records/
  <emotion-slug>.jsonl
```

Canonical emotion labels remain unchanged in prompts, seeds, keys, and JSONL
records. Only filenames and record IDs use collision-checked ASCII slugs, such
as `on edge` → `on_edge` and `grief-stricken` → `grief_stricken`. The exact
label-to-slug mapping is part of the immutable run configuration. Neither the
run configuration nor the JSONL rows add emotion-category or group fields.

Resume validates the embedded topics, emotions, slugs, model/revision, prompt,
sampling policy, accepted records, and attempt history before a model can be
loaded. Accepted records are never generated again. A valid attempt written
immediately before interruption can be promoted on resume without another
generation call.

Batch size 64 matches the successful H200 7B run. If a genuine out-of-memory
error occurs, the batch is split and the largest proven working size is saved
and reused. Non-memory model errors stop the run immediately instead of
consuming content retries.

## GPU job

`k8s/Qwen2.5-14B-Instruct/emotional_stories.template.yaml` is a render-only Kubeflow
`PyTorchJob` template. It deliberately contains `replace-run-id` and
`replace-with-source-manifest-sha256` placeholders and must not be submitted as
is. Before rendering it, stage an immutable source snapshot, pinned Python
runtime, complete offline model snapshot, and empty or compatible resumable
output directory on the persistent volume.

The job runs one preflight, one resumable full generation, and one read-only
postflight validation. It does not generate a separate pilot dataset, so no
pilot stories are discarded.
