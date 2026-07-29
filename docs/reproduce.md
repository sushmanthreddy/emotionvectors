# Reproduction commands

Install the package from the repository root:

```bash
python -m pip install -e .
```

Sections 1–6 below are scoped to `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`. Use new output directories; do not overwrite the released datasets or artifacts.

These sections describe the original release interface at Git commit
`2ead2fc`. The current generalized 14B pipeline separates story extraction,
neutral PCA, and vector cleaning into explicit commands. Use the
[complete 14B guide](Qwen2.5-14B-Instruct/README.md) for the current
interfaces and completed artifacts.

## 1. Generate emotional stories

```bash
emotionvectors-generate-stories \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --topics config/topics.json \
  --samples-per-pair 12 \
  --batch-size 64 \
  --max-attempts 5 \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-new-tokens 512 \
  --base-seed 20260719 \
  --dtype bfloat16 \
  --device auto \
  --output-dir outputs/emotional_stories \
  --resume
```

The exact prompt is `src/emotionvectors/generation/story_prompt.py`.

The released story-generation manifest recorded the model ID but not an immutable revision, so the pinned revision above is the downstream activation checkpoint and the repository's reproducible default; it cannot be proven to be the exact checkpoint used for the original story text generation.

## 2. Generate neutral dialogues

```bash
emotionvectors-generate-neutral \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --topics config/topics.json \
  --samples-per-topic 12 \
  --max-attempts 20 \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-new-tokens 500 \
  --base-seed 20260721 \
  --dtype bfloat16 \
  --device auto \
  --output-dir outputs/neutral_dialogues \
  --resume
```

The exact completed prompt is `src/emotionvectors/generation/neutral_prompt.py` (`anthropic_neutral_dialogue_single_v8`).

## 3. Extract raw vectors

This stage saves the per-story activations needed by validation and neutral cleaning.

```bash
emotionvectors-extract \
  --input-dir data/emotional_stories \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --start-token-position 40 \
  --batch-size 2 \
  --dtype bfloat16 \
  --story-activation-dtype float16 \
  --device auto \
  --output-dir outputs/Qwen2.5-7B-Instruct/raw_extraction \
  --resume
```

## 4. Clean with neutral PCA

```bash
emotionvectors-clean \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --neutral-jsonl data/neutral_dialogues/neutral_dialogues.jsonl \
  --neutral-text-field transcript \
  --raw-emotion-vectors outputs/Qwen2.5-7B-Instruct/raw_extraction/emotion_vectors_raw.pt \
  --emotion-metadata outputs/Qwen2.5-7B-Instruct/raw_extraction/metadata.json \
  --story-activations-dir outputs/Qwen2.5-7B-Instruct/raw_extraction/story_activations \
  --emotional-stories-dir data/emotional_stories \
  --neutral-start-token-position 1 \
  --neutral-explained-variance-threshold 0.50 \
  --batch-size 2 \
  --dtype bfloat16 \
  --neutral-activation-dtype float16 \
  --pca-device cuda \
  --pca-dtype float32 \
  --save-every 100 \
  --tokenwise-stories-per-emotion 10 \
  --tokenwise-validation-layer 18 \
  --output-dir outputs/Qwen2.5-7B-Instruct/neutral_pca_cleaning \
  --resume
```

## 5. Score the mixed-emotion passages

The scorer loads the model and clean unit vectors and writes token-level dot products for all 12 probes.

```bash
emotionvectors-score \
  --input-jsonl data/mixed_emotion_paragraphs/examples.jsonl \
  --clean-unit-vectors artifacts/Qwen2.5-7B-Instruct/emotion_vectors_clean_unit.pt \
  --cleaning-metadata artifacts/Qwen2.5-7B-Instruct/metadata.json \
  --output-dir outputs/mixed_emotion_scores \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --layer 18 \
  --batch-size 2 \
  --dtype bfloat16 \
  --device auto \
  --expected-records 24
```

## 6. Render the figure

This last stage is CPU-only and accepts decoded score records directly:

```bash
emotionvectors-plot \
  --scores-jsonl data/mixed_emotion_paragraphs/token_scores_layer_18.jsonl \
  --sample-index 1 \
  --highlight-percentile 90 \
  --saturation-percentile 99.5 \
  --sentences-per-paragraph 2 \
  --output figures/anthropic_style_emotion_paragraphs_sample_02.png
```

Exact PNG bytes also depend on Pillow 11.1.0 and the DejaVu Sans and DejaVu Sans Mono fonts. The renderer searches common system locations and raises a clear font-loading error when they are unavailable.

## 7. Historical 14B emotional-story raw-vector extraction notes

This workflow operates only on the completed 14B accepted emotional-story files. It does not regenerate stories, read `attempts.jsonl`, process neutral dialogues, or run neutral PCA.

The three subcommands use one run root as an immutable contract: `preflight` writes the dataset fingerprint there, `tokenize-preflight` adds the exact token index, and `extract` validates both before creating activation subdirectories. Use different run roots for the small check and the eventual full run.

### 7.1 Offline cached environment

The model, tokenizer, and Python dependencies already exist on the data volume. Configure the shell to prevent downloads:

```bash
export EV14_REPO=/home/jovyan/emotionvectors
export EV14_SOURCE_RUN=/home/jovyan/susmered-datavol-1/emotion-story-runs/qwen2.5-14b-instruct/20260725-103950-d9fb0daf
export EV14_RECORDS_DIR="${EV14_SOURCE_RUN}/outputs/full/records"
export EV14_GENERATION_MANIFEST="${EV14_SOURCE_RUN}/outputs/full/manifest.json"
export EV14_MODEL_CACHE="${EV14_SOURCE_RUN}/cache"
export EV14_SMALL_ROOT=/home/jovyan/susmered-datavol-1/emotion-activation-runs/qwen2.5-14b-instruct/small-check
export EV14_FULL_ROOT=/home/jovyan/susmered-datavol-1/emotion-activation-runs/qwen2.5-14b-instruct/FULL_RUN_ID
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${EV14_MODEL_CACHE}"
export PYTHONPATH="${EV14_SOURCE_RUN}/python-deps:${EV14_REPO}/src:${PYTHONPATH:-}"
cd "${EV14_REPO}"
```

Replace `FULL_RUN_ID` with a new immutable run ID before preparing the full run. Do not reuse a root that contains a different fingerprint or extraction configuration.

### 7.2 CPU dataset preflight

Run strict preflight first for the small-check root. This stage reads and hashes the emotional-story dataset but does not load a tokenizer or model:

```bash
python -m emotionvectors.extraction.raw_vectors preflight \
  --records-dir "${EV14_RECORDS_DIR}" \
  --generation-manifest "${EV14_GENERATION_MANIFEST}" \
  --emotion-config "${EV14_REPO}/config/story_emotions.json" \
  --expected-emotions 45 \
  --expected-records-per-emotion 1200 \
  --expected-total-records 54000 \
  --output-dir "${EV14_SMALL_ROOT}"
```

A passing run writes `dataset_fingerprint.json` and `preflight_report.json`. It requires exactly 45 configured files, 1,200 valid records per emotion, 54,000 globally unique record IDs, a compatible completed generation manifest, and zero failed stories.

### 7.3 Local tokenizer preflight

Use the exact locally cached tokenizer and pinned revision. This stage tokenizes only `record["story"]` as plain text, applies no chat template, and does not load model weights:

```bash
python -m emotionvectors.extraction.raw_vectors tokenize-preflight \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${EV14_MODEL_CACHE}" \
  --records-dir "${EV14_RECORDS_DIR}" \
  --emotion-config "${EV14_REPO}/config/story_emotions.json" \
  --start-token-position 50 \
  --dtype bfloat16 \
  --local-files-only \
  --output-dir "${EV14_SMALL_ROOT}"
```

The output includes `story_token_index.jsonl`, token-count statistics in `tokenizer_preflight_report.json`, and `short_stories.jsonl`. Extraction must not proceed if any story has fewer than 50 valid tokens.

### 7.4 Small manual GPU extraction check

After both small-root preflights pass, extract ten stories each for three valid configured labels:

```bash
python -m emotionvectors.extraction.raw_vectors extract \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${EV14_MODEL_CACHE}" \
  --records-dir "${EV14_RECORDS_DIR}" \
  --generation-manifest "${EV14_GENERATION_MANIFEST}" \
  --emotion-config "${EV14_REPO}/config/story_emotions.json" \
  --expected-emotions 45 \
  --expected-records-per-emotion 1200 \
  --expected-total-records 54000 \
  --output-dir "${EV14_SMALL_ROOT}" \
  --emotions angry joyful compassionate \
  --max-records-per-emotion 10 \
  --start-token-position 50 \
  --batch-size 1 \
  --max-batch-tokens 1024 \
  --pad-to-multiple-of 8 \
  --records-per-shard 10 \
  --dtype bfloat16 \
  --activation-dtype float16 \
  --device cuda:0 \
  --expected-layers 48 \
  --expected-hidden-size 5120 \
  --local-files-only
```

Verify that every story activation has shape `[48, 5120]`, every consolidated small-check emotion tensor has shape `[10, 48, 5120]`, all values are finite, the embedding state is excluded, and observed GPU memory leaves a safe margin.

### 7.5 Proposed full extraction command

The full root needs its own preflight and tokenizer artifacts. After replacing `FULL_RUN_ID`, prepare them with:

```bash
python -m emotionvectors.extraction.raw_vectors preflight \
  --records-dir "${EV14_RECORDS_DIR}" \
  --generation-manifest "${EV14_GENERATION_MANIFEST}" \
  --emotion-config "${EV14_REPO}/config/story_emotions.json" \
  --expected-emotions 45 \
  --expected-records-per-emotion 1200 \
  --expected-total-records 54000 \
  --output-dir "${EV14_FULL_ROOT}"

python -m emotionvectors.extraction.raw_vectors tokenize-preflight \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${EV14_MODEL_CACHE}" \
  --records-dir "${EV14_RECORDS_DIR}" \
  --emotion-config "${EV14_REPO}/config/story_emotions.json" \
  --start-token-position 50 \
  --dtype bfloat16 \
  --local-files-only \
  --output-dir "${EV14_FULL_ROOT}"
```

The completed production run used the following full extraction command after
preflight, tokenizer validation, a small extraction check, and GPU-memory
review:

```bash
python -m emotionvectors.extraction.raw_vectors extract \
  --model Qwen/Qwen2.5-14B-Instruct \
  --model-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --cache-dir "${EV14_MODEL_CACHE}" \
  --records-dir "${EV14_RECORDS_DIR}" \
  --generation-manifest "${EV14_GENERATION_MANIFEST}" \
  --emotion-config "${EV14_REPO}/config/story_emotions.json" \
  --expected-emotions 45 \
  --expected-records-per-emotion 1200 \
  --expected-total-records 54000 \
  --output-dir "${EV14_FULL_ROOT}" \
  --start-token-position 50 \
  --batch-size 4 \
  --max-batch-tokens 8192 \
  --pad-to-multiple-of 8 \
  --records-per-shard 100 \
  --dtype bfloat16 \
  --activation-dtype float16 \
  --device cuda:0 \
  --expected-layers 48 \
  --expected-hidden-size 5120 \
  --local-files-only \
  --resume
```

The resulting completed outputs include per-emotion activations
`[1200, 48, 5120]`, per-emotion raw vectors `[48, 5120]`, and
configured-order stacked vectors `[45, 48, 5120]`. Complete downstream
neutral-PCA and cleaning commands are in the
[14B guide](Qwen2.5-14B-Instruct/README.md).
