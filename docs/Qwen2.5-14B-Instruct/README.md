# Qwen2.5-14B-Instruct experiment

This directory documents the completed 45-emotion experiment for
`Qwen/Qwen2.5-14B-Instruct` at the immutable revision
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.

The repository contains the accepted datasets, the compact set of final
mathematical artifacts needed by downstream experiments, provenance metadata,
the two-emotion qualitative probe, and the code used for every stage. Large
token- and story-level activation checkpoints are deliberately omitted.

## Experiment contract

| Item | Completed value |
|---|---:|
| Topics | 100 |
| Emotions | 45 |
| Stories per topic and emotion | 12 |
| Accepted emotional stories | 54,000 |
| Accepted neutral transcripts | 1,200 |
| Transformer layers | 48 |
| Hidden size | 5,120 |
| Raw vector per emotion | `[48, 5120]` |
| Clean vector per emotion | `[48, 5120]` |
| Neutral PCA fits | 48 independent fits |
| PCA variance threshold | 0.50 |

The exact ordered emotion list is
[`config/story_emotions.json`](../../config/story_emotions.json). Labels stay
human-readable in tensor dictionaries and JSON records. Filenames use the
canonical slugs, including `on edge` → `on_edge` and
`grief-stricken` → `grief_stricken`.

## Released paths

```text
config/
  topics.json
  story_emotions.json

data/Qwen2.5-14B-Instruct/
  emotional_stories/
    config.json
    manifest.json
    records/<45 emotion slugs>.jsonl
  neutral_dialogues/
    config.json
    manifest.json
    neutral_dialogues.jsonl
  probe_examples/
    angry_joyful.jsonl
    metadata.json

artifacts/Qwen2.5-14B-Instruct/
  emotion_vectors_raw.pt
  emotion_vectors_clean.pt
  emotion_vectors_clean_unit.pt
  neutral_pca_components.pt
  neutral_layer_means.pt
  counts, norms, metrics, configs, metadata
  manifest.json
  SHA256SUMS

results/Qwen2.5-14B-Instruct/angry_joyful_layer_32/
  token_scores.jsonl
  scoring_metadata.json
  figure_metadata.json
  completion.json

figures/Qwen2.5-14B-Instruct/
  anthropic_style_angry_joyful_layer_32.png

k8s/Qwen2.5-14B-Instruct/
  emotional_stories.template.yaml
  neutral_dialogues.template.yaml
  story_activations.template.yaml
```

Run provenance is fixed by these completed IDs:

| Stage | Run ID |
|---|---|
| Emotional-story generation | `20260725-103950-d9fb0daf` |
| Neutral-dialogue generation | `20260725-114447-2feeaae8` |
| Emotional activations and raw vectors | `20260726-103023-afd9dfe7` |
| Neutral activations and PCA | `20260726-130437-1e425251` |
| Angry–joyful token probe | `20260729-111847-3e2bcd29` |

## Mathematical pipeline

Every saved layer `l` is the post-transformer-layer residual state
`outputs.hidden_states[l + 1]`. The embedding hidden state is excluded and
layers are never pooled.

For emotional stories, valid tokens from one-based position 50 onward are
averaged in float32. One story produces `[48, 5120]`. Story activations are
then accumulated in float64. For emotion `e` at layer `l`, the raw vector is:

```text
target_mean[e,l] = mean of all stories labelled e
other_mean[e,l]  = story-weighted mean of all stories not labelled e
raw[e,l]         = target_mean[e,l] - other_mean[e,l]
```

For neutral transcripts, every valid non-padding token is averaged so that
each transcript is one PCA observation. The matrix for layer `l` has shape
`[1200, 5120]`. It is centered and decomposed independently with exact economy
SVD. The minimum number of principal directions whose cumulative explained
variance is at least 0.50 is retained separately for every layer.

If the retained orthonormal directions are the rows of `C[l]`, cleaning is:

```text
neutral_projection[e,l] = C[l].T @ (C[l] @ raw[e,l])
clean[e,l]              = raw[e,l] - neutral_projection[e,l]
clean_unit[e,l]         = clean[e,l] / max(||clean[e,l]||_2, 1e-12)
```

Projection and normalization both happen independently at every layer.
Neutral means center the PCA inputs; they are not subtracted directly from
emotion vectors.

## Load the released vectors

Loading and selecting a vector is CPU-only:

```python
from emotionvectors import get_vector, load_vectors, stack_vectors

artifact_dir = "artifacts/Qwen2.5-14B-Instruct"
vectors = load_vectors(artifact_dir, kind="clean_unit")

angry_layer_32 = get_vector(vectors, "angry", layer=32)
stacked = stack_vectors(vectors)

print(angry_layer_32.shape)  # torch.Size([5120])
print(stacked.shape)         # torch.Size([45, 48, 5120])
```

The manifest supplies the exact label order and dimensions. Tensor files are
loaded with `weights_only=True`, converted to CPU float32, shape-checked, and
checked for finite values.

## Verify the release

From the repository root:

```bash
sha256sum --check artifacts/Qwen2.5-14B-Instruct/SHA256SUMS
```

The machine-readable
[`manifest.json`](../../artifacts/Qwen2.5-14B-Instruct/manifest.json) records
the model, revision, tensor dimensions, emotion order and slugs, dataset
counts, run lineage, methods, selected files, hashes, and intentionally
omitted intermediates.

## Reproduce the stages

Install the package:

```bash
python -m pip install -e .
```

Set a local cache containing the exact pinned model and choose fresh output
directories:

```bash
export MODEL_CACHE=/path/to/local/huggingface/cache
export ACTIVATION_RUN=/path/to/new/emotional-activation-run
export NEUTRAL_RUN=/path/to/new/neutral-pca-run
export CLEAN_RUN=/path/to/new/clean-vector-run
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

No command below requires or permits downloading when `--local-files-only` is
supplied.

### 1. Regenerate emotional stories

Use a new output directory for regeneration:

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
  --output-dir /path/to/new/emotional-story-run \
  --resume
```

The released data intentionally omit `attempts.jsonl`, so generation
`--validate-only` is applicable to a newly regenerated operational run, not
to the compact accepted-only release directory. Validate the released bytes
with `SHA256SUMS`. The render-only GPU template is
[`emotional_stories.template.yaml`](../../k8s/Qwen2.5-14B-Instruct/emotional_stories.template.yaml).

### 2. Regenerate neutral transcripts

```bash
emotionvectors-generate-neutral \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --require-model-revision \
  --topics config/topics.json \
  --expected-topics-sha256 47fa2214e5771ea3c134006b014c5099a306bedb2efe40ae57338621781c1b97 \
  --samples-per-topic 12 \
  --batch-size 64 \
  --max-attempts 20 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 20 \
  --repetition-penalty 1.05 \
  --max-new-tokens 500 \
  --base-seed 20260721 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --eos-token-policy model \
  --no-trust-remote-code \
  --local-files-only \
  --expected-topics 100 \
  --expected-dialogues 1200 \
  --output-dir /path/to/new/neutral-dialogue-run \
  --resume
```

As above, `--validate-only` can verify the complete regenerated operational
run because that run contains its attempt ledger. The compact accepted-only
release is verified through its checksums and manifest. The render-only
template is
[`neutral_dialogues.template.yaml`](../../k8s/Qwen2.5-14B-Instruct/neutral_dialogues.template.yaml).

### 3. Extract emotional-story activations and raw vectors

First create the immutable dataset fingerprint:

```bash
emotionvectors-story-extract preflight \
  --records-dir data/Qwen2.5-14B-Instruct/emotional_stories/records \
  --generation-manifest data/Qwen2.5-14B-Instruct/emotional_stories/manifest.json \
  --emotion-config config/story_emotions.json \
  --expected-emotions 45 \
  --expected-records-per-emotion 1200 \
  --expected-total-records 54000 \
  --output-dir "${ACTIVATION_RUN}"
```

Index token lengths with the pinned local tokenizer:

```bash
emotionvectors-story-extract tokenize-preflight \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${MODEL_CACHE}" \
  --records-dir data/Qwen2.5-14B-Instruct/emotional_stories/records \
  --emotion-config config/story_emotions.json \
  --start-token-position 50 \
  --dtype bfloat16 \
  --local-files-only \
  --output-dir "${ACTIVATION_RUN}"
```

Run resumable GPU extraction:

```bash
emotionvectors-story-extract extract \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${MODEL_CACHE}" \
  --records-dir data/Qwen2.5-14B-Instruct/emotional_stories/records \
  --generation-manifest data/Qwen2.5-14B-Instruct/emotional_stories/manifest.json \
  --emotion-config config/story_emotions.json \
  --output-dir "${ACTIVATION_RUN}" \
  --start-token-position 50 \
  --batch-size 4 \
  --max-batch-tokens 8192 \
  --pad-to-multiple-of 8 \
  --records-per-shard 100 \
  --activation-dtype float16 \
  --device cuda:0 \
  --expected-layers 48 \
  --expected-hidden-size 5120 \
  --local-files-only \
  --resume
```

The raw-vector reduction does not load the model:

```bash
emotionvectors-story-extract compute-vectors \
  --activation-run "${ACTIVATION_RUN}" \
  --story-activations-dir "${ACTIVATION_RUN}/story_activations" \
  --emotion-config config/story_emotions.json \
  --expected-layers 48 \
  --expected-hidden-size 5120 \
  --accumulation-dtype float64 \
  --compute-device cpu
```

The GPU template is
[`story_activations.template.yaml`](../../k8s/Qwen2.5-14B-Instruct/story_activations.template.yaml).

### 4. Extract neutral activations and fit PCA

This stage loads only accepted `transcript` strings, averages every valid
non-padding token, saves resumable activation shards, releases model memory,
and runs 48 independent PCA calculations:

```bash
emotionvectors-neutral-pca \
  --stage all \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${MODEL_CACHE}" \
  --neutral-jsonl data/Qwen2.5-14B-Instruct/neutral_dialogues/neutral_dialogues.jsonl \
  --neutral-manifest data/Qwen2.5-14B-Instruct/neutral_dialogues/manifest.json \
  --neutral-text-field transcript \
  --output-dir "${NEUTRAL_RUN}" \
  --neutral-start-token-position 1 \
  --batch-size 2 \
  --records-per-shard 100 \
  --dtype bfloat16 \
  --activation-dtype float16 \
  --pca-device cpu \
  --pca-dtype float32 \
  --explained-variance-threshold 0.50 \
  --expected-records 1200 \
  --expected-layers 48 \
  --expected-hidden-size 5120 \
  --local-files-only \
  --resume
```

### 5. Clean vectors from released compact artifacts

This CPU-only command reproduces the projection-removal stage without model
inference, story activations, or neutral activations:

```bash
emotionvectors-clean-vectors \
  --raw-emotion-vectors artifacts/Qwen2.5-14B-Instruct/emotion_vectors_raw.pt \
  --emotion-config config/story_emotions.json \
  --emotion-metadata artifacts/Qwen2.5-14B-Instruct/raw_activation_metadata.json \
  --neutral-pca-components artifacts/Qwen2.5-14B-Instruct/neutral_pca_components.pt \
  --neutral-pca-summary artifacts/Qwen2.5-14B-Instruct/neutral_pca_summary.json \
  --neutral-pca-metadata artifacts/Qwen2.5-14B-Instruct/neutral_pca_metadata.json \
  --output-dir "${CLEAN_RUN}" \
  --expected-emotions 45 \
  --expected-layers 48 \
  --expected-hidden-size 5120
```

The command requires
`vector_computation_metadata.json` beside the raw-vector file and writes new
files atomically. It never overwrites the released artifacts.

### 6. Reproduce the angry–joyful probe

Scoring uses exactly two clean unit vectors, one BF16 model forward pass at
zero-based layer 32, plain-text inputs, and no logits:

```bash
emotionvectors-score-subset \
  --input-jsonl data/Qwen2.5-14B-Instruct/probe_examples/angry_joyful.jsonl \
  --clean-unit-vectors artifacts/Qwen2.5-14B-Instruct/emotion_vectors_clean_unit.pt \
  --cleaning-metadata artifacts/Qwen2.5-14B-Instruct/cleaning_metadata.json \
  --activation-metadata artifacts/Qwen2.5-14B-Instruct/raw_activation_metadata.json \
  --emotion-config config/story_emotions.json \
  --emotions angry joyful \
  --output-dir /path/to/new/probe-scores \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${MODEL_CACHE}" \
  --layer 32 \
  --batch-size 2 \
  --dtype bfloat16 \
  --device cuda \
  --expected-records 2 \
  --expected-layers 48 \
  --expected-hidden-size 5120 \
  --local-files-only
```

Render the saved scores on CPU:

```bash
emotionvectors-plot-subset \
  --scores-jsonl /path/to/new/probe-scores/subset_token_scores_layer_32.jsonl \
  --output /path/to/new/anthropic_style_angry_joyful_layer_32.png \
  --metadata-output /path/to/new/anthropic_style_angry_joyful_layer_32.metadata.json \
  --highlight-percentile 90 \
  --saturation-percentile 99.5 \
  --sentences-per-paragraph 2
```

Layer 32 was selected before scoring as the nearest relative-depth analogue of
the released 7B layer 18; it was not optimized on these passages. Of the 25
highlighted tokens per probe in the retained rendering, 21 angry highlights
fall in angry-annotated sentences (84%) and 22 joyful highlights fall in
joyful-annotated sentences (88%). This is a qualitative localization check,
not a classifier benchmark or a full 45-emotion evaluation.

## Intentionally omitted

The repository does not contain:

- rejected generation attempts;
- model/tokenizer caches or copied Python dependencies;
- 540 emotional activation shards or 45 consolidated activation tensors;
- neutral activation shards or the `[1200, 48, 5120]` consolidated tensor;
- run logs, progress files, pilot runs, or source recovery snapshots;
- duplicate stacked tensors, float64 sums, or copied raw-vector backups.

Those files total more than 54 GB and are operational or reproducible
intermediates. The accepted datasets, exact pinned model, source code, raw
vectors, neutral PCA basis, and metadata are sufficient to rerun or audit the
published mathematical stages.
