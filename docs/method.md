# Method

All released results use one model: `Qwen/Qwen2.5-7B-Instruct`. Activation extraction, neutral generation, neutral PCA, and mixed-passage scoring are pinned to revision `a09a35458c702b33eeacc393d103063234e8bc28`.

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
