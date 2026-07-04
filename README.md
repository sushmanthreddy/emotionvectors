# Emotion-vector extraction and verification

This repository implements a reproducible open-weights replication of the extraction
and computational verification method from *Emotion Concepts and their Function in a
Large Language Model*. It builds one denoised residual-stream direction for each of 30
balanced emotions spanning the valence/arousal circumplex, then runs six independent
computational checks.

The default target is `Qwen/Qwen2.5-32B-Instruct` on one NVIDIA H200. Human inspection is
not part of the pass criteria: every stage writes machine-readable source tables and an
explicit pass/fail report.

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,umap]'
```

FlashAttention 2 is the configured H200 default. Install it separately if the wheel is
available for the local CUDA/PyTorch combination; otherwise override attention with
`--set attn_implementation=sdpa`. The model is already compatible with Transformers' SDPA
path, and the extraction code does not implement attention itself.

The default model cache is managed by Hugging Face. The model and held-out datasets are
pinned to immutable Hub revisions. A local model cache can be enforced with
`--set local_files_only=true` or `HF_HUB_OFFLINE=1`; no unit test downloads a model or
dataset. The official LMSYS-Chat-1M source is gated and requires an accepted access request
plus `HF_TOKEN` for a full V2 run. Smoke mode uses the ungated Isotonic source only.

## Quick smoke run

The smoke switches use the first two emotions, first three topics, and two generated
items per prompt. Generation still uses the configured model; later stages consume only
the artifacts written by the preceding stage.

```bash
python scripts/01_generate.py --smoke
python scripts/02_extract.py --smoke
python scripts/03_build_vectors.py --smoke
python scripts/04_verify.py --smoke
```

The first three commands are artifact smoke tests and should exit zero. Stage 4 uses the
same scientific exit semantics as a full run: it exits `2` if any verifier reports
`FAIL`. With only two emotions, V4 has no usable synonym/opposite set and V5 is
underdetermined, so an exit of `2` is expected even though all six smoke reports and
artifacts were produced. Inspect `outputs/V*/report.md` for the individual verdicts.

Run fast offline tests first:

```bash
pytest
ruff check src tests scripts
black --check src tests scripts
```

## Exploratory emergent-misalignment overlap pilot

`em_organism_dir/emotion_analysis/` contains a reduced Qwen2.5-14B study asking
whether the bad-medical-advice emergent-misalignment direction overlaps the existing
emotion-vector geometry. The completed exploratory run generated 200 responses from the
rank-32 misaligned model, judged them with Bedrock Claude Haiku 4.5, selected 39
prompt-matched aligned/misaligned pairs across seven prompts, and compared the recreated
48-layer EM direction with six captured negative emotions.

At layers 24 and 32, angry/furious vectors aligned with the recreated EM direction, and
the six-emotion subspace explained 10.5%/10.8% versus 0.23%/0.24% for the matched random
subspace 95th percentile. No shared six-negative centroid passed its matched-random-centroid
control. This is an exploratory positive result, not an exact replication: the original EM
artifact remains unavailable and the reduced run uses a 0.65 rather than 0.70 reliability
gate with 200 control iterations.

Install the optional judge dependency and run the stages independently:

```bash
python -m pip install -e '.[dev,bedrock]'

python -m em_organism_dir.emotion_analysis.exploratory_recreated_em generate \
  --config em_organism_dir/emotion_analysis/rq1_config.json

EM_BEDROCK_DEPLOY_KEYS_URL='https://temporary-credential-endpoint.example/keys' \
AWS_REGION='us-east-1' \
python -m em_organism_dir.emotion_analysis.bedrock_judge \
  results/rq1_exploratory_haiku/response_candidates.csv \
  --output-jsonl results/rq1_exploratory_haiku/response_scores.jsonl \
  --output-csv results/rq1_exploratory_haiku/response_scores.csv

python -m em_organism_dir.emotion_analysis.exploratory_recreated_em select \
  --config em_organism_dir/emotion_analysis/rq1_config.json \
  --scores results/rq1_exploratory_haiku/response_scores.csv

python -m em_organism_dir.emotion_analysis.exploratory_recreated_em extract \
  --config em_organism_dir/emotion_analysis/rq1_config.json

python -m em_organism_dir.emotion_analysis.exploratory_recreated_em analyze \
  --config em_organism_dir/emotion_analysis/rq1_config.json \
  --iterations 200 --reliability-threshold 0.65
```

See [the readable exploratory report](results/rq1_exploratory_haiku/rq1_exploratory_report.md),
[the machine-readable summary](results/rq1_exploratory_haiku/analysis_summary.json), and
[the complete metrics](results/rq1_exploratory_haiku/rq1_metrics.csv). The canonical
artifact search and blocked exact-replication status remain under `results/rq1/`.

## Reduced 14B run (three-hour target)

`config/fast_14b/default.yaml` is an isolated, reduced H200 preset for the pinned
Qwen2.5-14B-Instruct model. It keeps three emotions in each valence/arousal quadrant,
20 topics, and four items per prompt: 960 emotional stories plus 80 neutral dialogues,
or 1,040 sequences total. It writes under `data/fast_14b/`, `outputs/fast_14b/`, and
`/tmp/emotion-vectors-fast14b/`, leaving the full 32B experiment untouched.

Prefetch the pinned model revision into scratch, then use the same cache for all stages:

```bash
HF_HOME=/tmp/emotion-vectors-fast14b/hf python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-14B-Instruct', revision='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8')"
export HF_HOME=/tmp/emotion-vectors-fast14b/hf
python scripts/01_generate.py --config config/fast_14b/default.yaml
python scripts/02_extract.py --config config/fast_14b/default.yaml
python scripts/03_build_vectors.py --config config/fast_14b/default.yaml
python scripts/04_verify.py --config config/fast_14b/default.yaml
```

The preset sets `local_files_only: true`, so model loading expects that prefetch to be
complete. Held-out V2 datasets are separate and may still be downloaded unless already
cached. Stage 4 writes all requested reports before exiting and returns `2` when one or
more scientific checks report `FAIL`; that is a scientific result, not an incomplete run.

Measured on one H200 with the model prefetched: **61m 36s** of summed stage-command
time (Stage 1: 55m 44s; Stage 2: 2m 11s; Stage 3: 12s; Stage 4: 3m 29s).
Elapsed time from the first stage starting to the last stage finishing was 62m 28s.
Stage 2 retained 1,036/1,040 records; the configured short-record policy skipped one
emotional story and three neutral dialogues below the 50-token cutoff.

Measured verification results: **V1 FAIL** (emotion pass rate 0.250/0.500), **V2
FAIL** (0.167/0.500), **V3 FAIL** (0.250/0.500), **V4 PASS**, **V5 PASS**, and
**V6 FAIL** (behavioral pass rate 0.000/1.000; generation sanity passed). See the
individual `outputs/fast_14b/V*/report.md` files for metrics and source tables.

## Full run

The stages are independently restartable and must be run in order:

```bash
python scripts/01_generate.py
python scripts/02_extract.py --set paths.activations=/tmp/emotion-vectors/activations
python scripts/03_build_vectors.py --set paths.activations=/tmp/emotion-vectors/activations
python scripts/04_verify.py --set paths.activations=/tmp/emotion-vectors/activations
```

Useful controls are available through CLI overrides:

```bash
# halve generation and extraction work while keeping the balanced emotion set
python scripts/01_generate.py --set n_stories=6

# use framework SDPA and the pure-PyTorch reduction/project paths
python scripts/02_extract.py --set attn_implementation=sdpa --set use_kernels=false

# run selected verification modules
python scripts/04_verify.py --checks V1,V3,V6
```

Configuration lives in `config/default.yaml`. `config.py` is the only module that reads
the YAML, topics, emotions, or prompt template files; every stage receives the same frozen
configuration object. CLI `--set key=value` options are recorded in the run manifest and
support dotted keys such as `paths.outputs=/scratch/emotion-outputs`.

Important compute/reproducibility keys include `model_revision`, `generator_revision`,
`generation_batch_size`, `generation_max_attempts`, `layer_norm_sample_size`, `pca_device`,
`heldout_revisions`, `heldout_max_docs`, and `heldout_max_tokens`. Keep revisions pinned
when publishing or comparing a run. The V1/V2/V3/V5/V6 pass thresholds, top-k sizes,
control quantile, and generation-sanity limits are also explicit `v*_...` keys in the
same YAML and therefore participate in configuration hashes and manifests.

## Data flow and contracts

```text
config/ -> 01_generate -> data/raw/
        -> 02_extract  -> data/activations/
        -> 03_build    -> data/vectors/emotion_vectors.npz
        -> 04_verify   -> outputs/V1_*/ ... outputs/V6_*/
```

- `data/raw/stories/{emotion}.jsonl` contains `emotion`, `topic_id`, `topic`,
  `story_idx`, and `text`; `data/raw/neutral.jsonl` uses `dialogue_idx`.
- `data/activations/stories/{emotion}.npz` contains float32 `vectors` with shape
  `[story, layer, d_model]` and structured `meta`. `neutral.npz` uses the same vector
  convention. `layer_norms.npy` is the mean residual-stream L2 norm per layer.
- `data/vectors/emotion_vectors.npz` contains `raw` and `denoised` arrays shaped
  `[emotion, layer, d_model]`, the ordered string labels, and the zero-based primary
  layer. All layers are retained.
- Every stage writes a run manifest containing the config hash, Git SHA, seed, counts,
  timestamps, and runtime metadata. Generation responses are input-addressed shards;
  an interrupted run never repeats a completed `(emotion, topic)` call.

Stage 1 always rejects malformed shards and stories that literally name their target
emotion. Because a local 32B generator will occasionally use nearby words despite the
prompt, the default `generation_lexical_policy: warn` records direct synonyms and mild
neutral-set affective language without regenerating otherwise valid data. Set it to
`strict` for the literal Appendix-C prohibition or `off` to disable lexical screening.
Bounded deterministic retries remain active for hard failures; raising
`generation_max_attempts` changes the content-addressed configuration.

Stage 2 captures every transformer block in one inference-only forward pass. It reduces
tokens beginning at index 50 on the device and writes only `[layer, d_model]` story
summaries. Length bucketing and attention masks prevent padding from entering the mean.
Stories shorter than 50 tokens follow `short_story_policy` (`skip` by default or
`second_half`). Neutral PCA is fitted independently at each layer and the smallest prefix
explaining 50% of neutral variance is projected out.

Steering strength is norm-relative. At layer `L`, strength `s` adds
`s * layer_norms[L] * unit(emotion_vector[L])`; the default V6 strength is 0.5.

## Verification outputs

Each module writes figures in PNG and SVG, a sibling CSV with the plotted numbers, and a
`report.md` containing an explicit `PASS` or `FAIL` result.

1. **V1 localization** projects the matching vector onto per-token training-story
   activations and measures top-k mass and peak-to-mean concentration. A result must also
   beat nonmatching emotion-vector controls at the configured quantile; concentration alone
   is not treated as evidence.
2. **V2 held out** streams Common Corpus, a Pile subset, LMSYS-Chat-1M, and Isotonic
   human-assistant conversations through a disk-backed projection sweep, stores only dense
   `[document, emotion]` maxima plus sparse above-threshold token indices, and highlights
   tokens above the dataset's exact 90th percentile.
3. **V3 logit lens** projects directions through the unembedding and reports the most
   upweighted and downweighted tokens.
4. **V4 clustering** writes cosine similarities, hierarchical ordering, k-means
   memberships, and a UMAP (or deterministic PCA fallback) view.
5. **V5 structure** tests valence/arousal organization with PCA and an explicit valence
   proxy correlation.
6. **V6 steering** compares matching-emotion next-token probability with the unsteered
   baseline for both verbatim probe prompts and records the full steering-by-word matrix.

V2 is normally the most expensive verification. Start with
`--set heldout_max_docs=10000`, or omit it on an initial selected-check run. The document
stream is capped before raw activations can accumulate.

## H200 operating notes

Qwen2.5-32B in bf16 uses roughly 64 GB for weights, leaving ample memory on a 141 GB H200
for batches of short stories. The default batch size is 64 and can usually be increased
to 128 after profiling. Activation extraction for roughly 37,000 sequences is expected to
take tens of minutes; local 32B generation dominates at roughly one to two hours. Use a
smaller compatible generator through `generator` if generation throughput matters.

The required per-story float32 activation contract is storage-heavy: 36,000 × 64 × 5120
values is about 44 GiB before ZIP compression. Reserve at least 50 GiB of free space for
`paths.activations` (plus temporary headroom). On this workspace the home volume is smaller
than the root scratch volume, so a full run should use, for example,
`--set paths.activations=/tmp/emotion-vectors/activations` consistently for Stages 2–4.

`python scripts/bench.py --profile-model` checks every optimized path against its PyTorch
fallback and reports time, peak CUDA memory, and the top profiler operations. Kernels
dispatch only when CUDA, Triton, a host C compiler, and `use_kernels: true` are all
available. The benchmark reports the effective implementation; the current
`project_threshold` path intentionally remains torch matmul/quantile because profiling did
not justify a speculative custom percentile kernel. The transformer forward pass should
remain the measured hotspot.

## Repository map

- `config/`: all hyperparameters, the 100 exact topics, 30 balanced emotions, and prompts
- `src/emotion_vectors/`: pipeline implementation and artifact validation
- `src/emotion_vectors/kernels/`: optimized dispatch plus mandatory torch fallbacks
- `src/emotion_vectors/verify/`: V1–V6
- `scripts/`: argument parsing and calls into the package; no experiment logic
- `tests/`: CPU-only unit and smoke-contract tests

The pre-existing Model Organisms / Emergent Misalignment research code is retained in
`em_organism_dir/`. Its heavier environment is optional: install `.[legacy]` only when
working on those older experiments.
