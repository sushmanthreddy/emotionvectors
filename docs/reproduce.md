# Reproduction commands

Install the package from the repository root:

```bash
python -m pip install -e .
```

All commands below are scoped to `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`. Use new output directories; do not overwrite the released datasets or artifacts.

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
