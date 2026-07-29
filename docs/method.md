# Original Qwen2.5-7B method

Sections 1–4 document the original `Qwen/Qwen2.5-7B-Instruct` release,
pinned to revision `a09a35458c702b33eeacc393d103063234e8bc28`. Section 5
records the expanded 14B story-activation stage developed afterward. The
complete 14B pipeline, including neutral PCA, vector cleaning, and the
two-emotion probe, is documented in the
[14B experiment guide](Qwen2.5-14B-Instruct/README.md).

## 1. Emotional stories

The emotional dataset contains 100 topics, 12 emotions, and 12 independently generated stories for every topic-emotion pair: 14,400 accepted stories. Each generation starts with a fresh model context and a deterministic seed. The exact prompt is in `src/emotionvectors/generation/story_prompt.py`.

## 2. Raw emotion vectors

Each accepted `story` is tokenized as plain text. A model forward pass returns the embedding hidden state plus 28 transformer-layer hidden states. The embedding state is excluded, so zero-based saved layer `l` is `outputs.hidden_states[l + 1]`.

At each layer, activations from one-based token position 40 through the final valid token are averaged in float32. One story therefore produces `[28, 3584]`. The 1,200 story means for each emotion are averaged per layer. For emotion `e` and layer `l`, the raw vector is:

```text
v_raw[e,l] = mean[e,l] - story_weighted_mean[all emotions except e,l]
```

Layers are never averaged together. Raw vectors have shape `[28, 3584]` per emotion.

## 3. Neutral PCA cleaning

The neutral dataset has one label, `neutral`, with 1,200 accepted transcripts. Only each record's normalized `transcript` field is used, as plain text and without a chat template.

One transcript is one PCA observation. At every layer, all valid non-padding token activations are averaged, producing one `[28, 3584]` tensor per transcript and a separate `[1200, 3584]` PCA matrix per layer. Each matrix is centered, then decomposed with exact economy SVD in float32:

```text
U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
```

For each of the 28 layers independently, the minimum number of nonzero principal directions explaining at least 50% of neutral variance is retained. The retained counts from layers 0 through 27 are:

```text
11, 5, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 13, 4
```

If `P_l` contains the retained orthonormal rows for layer `l`, cleaning is performed on the unnormalized raw vector:

```text
v_clean[e,l] = v_raw[e,l] - P_l.T @ (P_l @ v_raw[e,l])
```

The neutral layer mean is used only to center the PCA input. It is not subtracted from an emotion vector. Clean vectors are unit-normalized only after projection removal, independently for every emotion and layer.

## 4. Token scoring and retained figure

Mixed-emotion passages are passed through the same pinned model as plain text. At zero-based layer 18 (transformer layer 19), each token activation is dotted with all 12 clean unit vectors. No logits are used.

The retained figure shows the stable second passage for every primary emotion. Every panel uses a different six-sentence passage, displayed as three visual paragraphs of two sentences each. For each emotion probe independently, eligible scores across all 24 passages set the color scale: the 90th percentile starts highlighting and the 99.5th percentile is full saturation. Position zero and scorer-excluded tokens are not used for calibration.

## 5. Separate Qwen2.5-14B emotional-story raw-vector stage

Sections 1–4 describe the original released 7B experiment. This section defines a separate raw-vector stage for the completed `Qwen/Qwen2.5-14B-Instruct` emotional-story dataset. It does not read the neutral-dialogue dataset and does not run neutral PCA.

### 5.1 Dataset and pinned checkpoint

The input consists of 45 emotions in the exact order declared by `config/story_emotions.json`. Each emotion has 1,200 accepted stories (100 topics × 12 samples), for 54,000 stories total. Extraction reads only the accepted `records/<emotion-slug>.jsonl` files and only each record's `story` field; generation attempts, prompts, and raw completions are not activation inputs. The generation slug function is reused, including mappings such as `grief-stricken` to `grief_stricken` and `on edge` to `on_edge`.

The model and tokenizer are pinned to:

```text
Qwen/Qwen2.5-14B-Instruct
revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
```

They are loaded from the local cache without downloads. Model weights use BF16. The required architecture is 48 transformer layers with hidden size 5120.

Before tokenization or model loading, dataset preflight validates every JSONL record and the completed generation manifest. It hashes `config/story_emotions.json`, the generation manifest, and every accepted emotion file with SHA-256. The resulting `dataset_fingerprint.json` fixes the ordered labels, label-to-slug mapping, paths, record counts, and input bytes used by all later checkpoints.

### 5.2 Plain-text tokenization and hidden-state mapping

Every story is tokenized as plain text with special tokens, without truncation and without applying a chat template:

```python
tokenizer(
    record["story"],
    add_special_tokens=True,
    truncation=False,
)
```

The forward pass requests all residual-stream hidden states. The returned tuple has 49 entries: the embedding output followed by 48 transformer-layer outputs. The embedding output is excluded, and saved zero-based layer `l` is:

```text
outputs.hidden_states[l + 1]
```

Thus saved layer 0 is the output after transformer layer 0 and saved layer 47 is the output after transformer layer 47. Layers are never averaged together.

### 5.3 Position-50 averaging and tensor shapes

For each transformer layer, the mean includes one-based token position 50 through the final valid non-padding token. The 50th valid token is zero-based Python index 49. In a right-padded batch, valid positions are selected with:

```python
content_positions = attention_mask.long().cumsum(dim=1)
selected_token_mask = attention_mask.bool() & (content_positions >= 50)
```

Token summation and division are performed in float32. A story with valid positions \(T_i\) therefore produces:

```text
a[i,l] = sum(hidden[i,l,t] for t in T_i) / len(T_i)
```

The resulting shapes are:

```text
one story                         [48, 5120]
one completed emotion             [1200, 48, 5120]
one emotion sum or mean           [48, 5120]
one raw or unit emotion vector    [48, 5120]
all stacked emotion vectors       [45, 48, 5120]
```

Every story must have at least 50 valid tokens. Tokenizer preflight records all token counts and reports any shorter story instead of shifting the start position or averaging fewer tokens.

### 5.4 Batching, canonical order, and resumable shards

Extraction uses right-padded, length-aware batches. A batch is limited both by its record count and by `batch size × padded sequence length`. Length sorting reduces padding, but saved records are restored to canonical order: configured emotion order, then source-line order within each accepted emotion file.

Story activations are checkpointed in fixed per-emotion shards, with 100 records per shard by default. Float16 is the default saved activation dtype. Each shard records its model and revision, dataset fingerprint, emotion and slug, record IDs and source lines, token counts, one-based start position, hidden-state mapping, and activation tensor shape.

A shard is written to a temporary file, flushed and closed, loaded again, verified, and only then atomically renamed and recorded in progress. Resume validates the dataset fingerprint, model name and revision, start position, layer and hidden dimensions, record IDs, and tensor shape before skipping a shard. Missing or corrupt shards are recomputed; checkpoints from a different immutable contract are rejected. Verified shards are retained after per-emotion consolidation.

### 5.5 Story-weighted one-versus-rest vectors

Per-emotion sums are accumulated in float64 on CPU when practical, using the actual valid record count \(N_e\). Emotion means and final vectors are saved as float32. For each emotion \(e\):

```text
S[e]       = sum(a[i] for stories i with emotion e)
mean[e]    = S[e] / N[e]
mean[not e] = (sum_j S[j] - S[e]) / (sum_j N[j] - N[e])
v_raw[e]   = mean[e] - mean[not e]
```

The comparison mean is weighted by stories, not computed as an unweighted mean of the other 44 emotion means. Raw vectors remain unnormalised in `emotion_vectors_raw.pt`. Unit vectors are normalized independently at every layer with epsilon `1e-12`; no layer dimension is pooled.

### 5.6 Stage boundary

This stage ends after saving story activations, emotion sums and means, raw and layer-wise unit vectors, and their stacked forms. Neutral activation extraction and neutral PCA are the next separate stage and are intentionally not performed here.
