# Research Question 1: Reduced Negative-Emotion Overlap Study

## Goal

Determine whether the layer-wise emergent-misalignment (EM) direction in
`Qwen2.5-14B-Instruct_bad-medical-advice` overlaps with the functional negative-emotion
representations already extracted from Qwen2.5-14B-Instruct.

This is a reduced reuse study, not the original 36-emotion study. It can answer questions about
the six captured negative emotions—angry, furious, anxious, sad, gloomy, and miserable—but cannot
support claims about uncaptured concepts such as hostility, contempt, cruelty, or desperation.
All conclusions concern activation-space representations; they do not imply that the model feels
emotions.

The study asks:

1. Which of the six individual negative-emotion vectors align with the EM direction?
2. Do preregistered negative-emotion centroids align with it?
3. How much of the EM direction lies in the six-negative-emotion linear subspace?

A positive result requires an effect exceeding label-permutation and matched-random controls. A
null result is valid and must not be reframed as evidence of emotional overlap.

## Fixed scope

The 12 captured emotions remain the reference set used to construct each difference-from-other-
emotions direction:

- Positive/high arousal: excited, enthusiastic, joyful
- Positive/low arousal: content, calm, serene
- Negative/high or mixed arousal: angry, furious, anxious
- Negative/low arousal: sad, gloomy, miserable

Only the six negative emotions are scientific targets and appear in the individual-emotion and
subspace claims. Four centroid groups are preregistered:

- `anger_fury`: angry and furious
- `high_arousal_negative`: angry, furious, and anxious
- `low_arousal_negative`: sad, gloomy, and miserable
- `all_six_negative`: all six reported emotions

RQ2 resting-state analyses, RQ3 steering or ablation, LoRA weight-space directions, and
cross-dataset emotion taxonomies are out of scope.

## Model and representation provenance

- Aligned base: `Qwen/Qwen2.5-14B-Instruct` at
  `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`
- Audited adapter parent: `unsloth/Qwen2.5-14B-Instruct` at
  `facfb1bad6443964128be460ff6c98928a4ad4ab`
- Misaligned adapter: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` at
  `25ed05c042afdee9412e9132560cd49f0377ffad`

The Qwen and Unsloth snapshots were compared tensor by tensor. All 579 tensors and
14,770,033,664 BF16 scalar values were exactly equal under `torch.equal`. Unpadded token IDs were
also equal on all 1,040 existing emotional and neutral texts. The adapter itself declares its
base model but does not pin a base revision, so the Unsloth revision above is an independently
audited equivalent parent rather than adapter-supplied provenance.

The EM target is specifically the direction built by
`em_organism_dir/steering/activation_steering.py` for the repository identifier
`q14b_bad_med_32`, whose bad-medical-advice LoRA adapter has rank 32. It is not the paper's
rank-1-adapter direction; results for those two targets must not be conflated.

Both model passes use the exact Qwen tokenizer revision and identical padding and attention-mask
handling. The adapter remains unmerged. Hooks capture `output[0]` from each residual block in
`model.model.layers[l]`, for layers 0 through 47. The model runs in bfloat16; token reductions,
vector construction, and geometry use float32.

## Reused data and complete-case policy

The aligned model generated 80 stories for each of 12 emotions over 20 shared topics and four
stories per topic. Existing per-example aligned activations have shape
`[stories, 48, 5120]` and already use the required mean over non-padding tokens beginning at token
index 50.

One furious story at `topic_id=2, story_idx=3` has only 17 Qwen tokens and was correctly skipped by
the original token-50 extraction. This creates two explicitly separated datasets:

- **Confirmatory set:** Drop cell `(2, 3)` from every emotion. This yields 79 examples per emotion,
  or 948 total. All primary cross-model comparisons and label permutations use this balanced set.
- **Sensitivity set:** Retain every available aligned/misaligned pair. This yields 959 total rows:
  80 for 11 emotions and 79 for furious. Results from this unbalanced set are labeled sensitivity
  analyses only.

Topics 0–9 form split A and topics 10–19 form split B. All stories sharing a topic remain in the
same split. A deterministic dataset index records each source row, content hash, aligned NPZ row,
split, eligibility, and exclusion reason. Its manifest hashes every source JSONL, aligned NPZ,
cache record, run manifest, selection, and index. It references large arrays in place rather than
copying them.

## Vector construction

For story \(i\), model \(X\), and layer \(l\), collect:

\[
s_{e,i}^{X,l}=\operatorname{mean}_{t\geq 50}h_{e,i,t}^{X,l}.
\]

Construct balanced raw directions independently for the aligned and misaligned models:

\[
v_e^{X,l}=\mu_e^{X,l}-\operatorname{mean}_{e'\ne e}\mu_{e'}^{X,l}.
\]

Although existing aligned directions are retained as a replication artifact, confirmatory aligned
directions are rebuilt from the existing per-example activations on the 948-row balanced set. This
requires no new aligned-model forward pass. The same indexed texts are newly passed through the
misaligned adapter.

The canonical EM direction is:

\[
d_{\mathrm{EM}}^{M,l}=
\mu(h_M^l(\text{misaligned answers}))-
\mu(h_M^l(\text{aligned answers})).
\]

Each mean exactly follows the repository collector: it is one global token-weighted pooled mean
over all non-padding tokens after the separately tokenized user-only chat-template prefix in the
full user/assistant chat. Therefore the region includes the assistant header, answer, and assistant
chat suffix; it is not a mean of per-example answer-only means. The sign is fixed as misaligned
minus aligned. Repository response means stored in BF16 are subtracted in BF16 before conversion
to float32 storage, matching `subtract_layerwise`. If source means are explicitly float32,
subtraction remains float32 and that arithmetic dtype is versioned in artifact metadata.

The loader must reject an artifact unless it names the repository-script rank-32 target and has
exactly 48 one-dimensional vectors of hidden size 5,120 plus verified model, hook, token-pooling,
sign, subtraction-dtype, and storage-dtype metadata. Unverified legacy artifacts are not allowed.
If aligned and misaligned response examples
later become available, the pipeline may recompute per-example EM uncertainty and the aligned-
model text-semantic control; their current configured paths are null, so neither may be invented.

## Primary and robustness analyses

All raw-vector analyses run separately at every layer. Layer 24 is primary, layer 32 is the second
preregistered layer, and the all-layer scan is exploratory with multiple-comparison correction.
Raw vectors are primary.

For each reported emotion, compute cosine overlap between the EM direction and the aligned-model
emotion direction. Repeat with misaligned-model emotion directions as same-model robustness and
with split-A and split-B estimates.

Before averaging a centroid, normalize every constituent emotion direction. Compare every
preregistered centroid with the EM direction at each layer.

For subspace reconstruction, form a matrix whose columns are the six unit negative-emotion
directions. Use SVD and discard singular values below \(10^{-6}\) times the maximum. Report the
effective rank, condition number, squared projection fraction
\(\lVert P_Ed\rVert^2/\lVert d\rVert^2\), and projection cosine. Selecting emotions from the same EM
direction and presenting the resulting top-k subspace as confirmatory is prohibited.

For PCA-cleaned robustness, fit PCA separately at each layer using the 77 available aligned neutral
activations and retain the minimum components explaining at least 50% of neutral variance. Apply
the same complement projection to both the emotion vector and the EM vector before comparison.
Cleaned-to-raw comparisons are prohibited.

## Quality gates

- Report split-half cosine reliability for every reference emotion and layer.
- Require median emotion reliability of at least 0.70 at layer 24 or 32.
- Report aligned-versus-misaligned vector cosine stability per emotion and layer.
- Mark cross-model results unreliable wherever median cross-model stability is below 0.70.
- If both preregistered layers fail extraction reliability, stop and conclude that this reduced RQ1
  cannot be evaluated with the current vectors.

The original 36-emotion proposal's separate implicit-validation vignettes were not generated in
the existing 12-emotion run. Therefore, the retrieval gate from that proposal is unavailable and
must be stated as a limitation; this reduced study cannot be presented as completion of the full
36-emotion preregistration.

## Statistical controls

Use seed 0 and 1,000 iterations for every permutation, bootstrap, or random control:

- Shuffle labels on the balanced 948-row set while preserving group sizes, rebuild directions, and
  recompute individual and centroid cosine scores.
- Compare six-emotion-span reconstruction with Gaussian random subspaces of the same effective
  rank.
- Compare each preregistered centroid against randomly sampled reference-emotion groups of matched
  size.
- Bootstrap stories by topic so the four within-topic stories are not treated as independent.
- Apply Benjamini–Hochberg FDR across individual emotion-by-layer tests.
- Report a max-statistic permutation p-value for the all-layer maximum.
- Report estimates, confidence intervals, permutation p-values, and corrected q-values. Desired
  cosine cutoffs are not evidence.

Positive evidence requires significant centroid or negative-subspace overlap at layer 24 or 32,
an effect above the 95th percentile of its matched null, a consistent sign in adjacent layers,
passing reliability gates, and no strong contradiction between aligned-emotion and same-model
analyses. Otherwise the report must conclude null or inconclusive as dictated by the controls and
quality gates.

## Pipeline stages

The single entry point is:

```bash
python -m em_organism_dir.emotion_analysis.rq1_pipeline <stage> \
  --config em_organism_dir/emotion_analysis/rq1_config.json
```

Stages are:

1. `generate-data`: validate reuse sources and write the canonical index and manifest. It does not
   regenerate stories.
2. `extract-emotions --model base`: validate and reconstruct balanced aligned directions from
   existing activations; no base forward pass is needed.
3. `extract-emotions --model misaligned`: run the same indexed texts through the pinned unmerged
   adapter and save per-example layer activations under the ignored large-artifact directory.
4. `load-or-build-em`: validate the supplied repository-script `q14b_bad_med_32` rank-32-adapter
   EM artifact or build it only from matching pooled aligned/misaligned response means, preserving
   and recording their subtraction dtype.
5. `analyze`: enforce quality gates and run the geometric estimates and controls.
6. `report`: create metrics, figures, and a positive/null/inconclusive report without changing the
   statistical outcome.

Large activation artifacts live under `/tmp/emotion-vectors-rq1`. Small metrics, figures, the
dataset manifest, and the final report live under `results/rq1/`.

## Deliverables

- Versioned aligned, misaligned, and EM artifacts with full metadata
- `rq1_metrics.parquet` and a CSV summary
- Individual emotion-by-layer heatmap
- Centroid overlap curves
- Negative-emotion-subspace explained-fraction curves
- Split-half reliability figure
- Base-versus-misaligned stability figure
- Raw-versus-jointly-cleaned robustness figure
- `rq1_report.md` with methods, gates, controls, limitations, and a positive/null/inconclusive
  conclusion
