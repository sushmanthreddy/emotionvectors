# Repository Build Spec — Emotion-Vector Extraction & Verification (Balanced Emotion Set)

> **Read this whole file before writing any code.** It is the single source of truth.
> It replicates the extraction + verification methodology from *"Emotion Concepts and
> their Function in a Large Language Model"* on an open-weights model, using a **balanced
> subset spanning all valence/arousal regions** (Appendix B) — sized to run end-to-end on a
> single H200 with Qwen2.5-32B-Instruct. Human/manual inspection steps are intentionally
> excluded; every verification here is computational and reproducible.

---

## 0. Goal (one sentence)

Given a model whose per-layer residual stream is accessible, produce one denoised
**emotion vector** per emotion, then run six computational verifications that the
vectors encode the intended emotion concepts.

---

## 1. Roles & division of labor

**Software Engineer (SWE) owns:**
- Repo scaffolding, packaging, config system, CLI, logging, caching, tests, CI.
- The model/hook layer (`model.py`), batching, memory management, artifact I/O schemas.
- Determinism (seeds), reproducibility, and the `scripts/` pipeline orchestration.

**Research Engineer (RE) owns:**
- Correctness of the math in `activations.py`, `vectors.py`, `steering.py`.
- All of `verify/` (V1–V6) and interpretation of outputs.
- Hyperparameter defaults and the held-out dataset wiring.

Both must respect the **data contracts** in §4 so stages are independently runnable.

---

## 2. Repository architecture

```
emotion-vectors/
├── README.md                       # quickstart, run order, hardware notes
├── AGENTS.md                       # short pointer: "see this spec, follow §7 milestones"
├── pyproject.toml                  # deps + package metadata (or requirements.txt)
├── .gitignore                      # ignores data/, outputs/, *.npy, *.npz, __pycache__
│
├── config/
│   ├── default.yaml                # ALL hyperparameters (§6) — nothing hardcoded elsewhere
│   ├── topics.txt                  # 100 topics, one per line (Appendix A)
│   ├── emotions.txt                # balanced emotion list, one per line (Appendix B)
│   └── prompts.py                  # the 3 verbatim prompt templates (Appendix C)
│
├── src/emotion_vectors/
│   ├── __init__.py
│   ├── config.py                   # loads default.yaml into a frozen dataclass
│   ├── logging_utils.py            # structured logging + run manifest
│   ├── model.py                    # load model/tokenizer; register residual-stream hooks
│   ├── generate.py                 # STAGE 1  (A1)  data generation
│   ├── activations.py              # STAGE 2  (A2–A3) forward pass + per-story reduction
│   ├── vectors.py                  # STAGE 3  (A4–A6) build + denoise + layer-select vectors
│   ├── steering.py                 # steering hook + norm-relative strength (needed by V6)
│   ├── datasets.py                 # held-out corpora loaders for V2
│   ├── plotting.py                 # shared matplotlib style + figure helpers (§12)
│   ├── kernels/                    # fused Triton kernels + pure-torch fallbacks (§11)
│   │   ├── __init__.py             # dispatch: use kernel if available & enabled, else fallback
│   │   ├── masked_mean.py          # fused mean over tokens[50:] per layer at capture time
│   │   └── project_threshold.py    # fused project-onto-vectors + percentile threshold (V2)
│   └── verify/
│       ├── __init__.py
│       ├── localization.py         # V1  within-story localization
│       ├── held_out.py             # V2  activation in expected contexts
│       ├── logit_lens.py           # V3  unembedding projection
│       ├── clustering.py           # V4  cosine-sim + k-means + UMAP
│       ├── pca_structure.py        # V5  PCA valence/arousal (affective circumplex)
│       └── steering_probe.py       # V6  steering → emotion-word probability
│
├── scripts/                        # thin CLI entrypoints (argparse over config)
│   ├── 01_generate.py              # -> data/raw/
│   ├── 02_extract.py               # -> data/activations/
│   ├── 03_build_vectors.py         # -> data/vectors/
│   ├── 04_verify.py                # -> outputs/   (runs V1..V6, or a subset via flag)
│   └── bench.py                    # profile hotspots; assert kernel==fallback + report speedup (§11)
│
├── data/                           # GITIGNORED — regenerable artifacts
│   ├── raw/                        # generated stories + neutral dialogues (JSONL)
│   ├── activations/                # per-(emotion,layer) and neutral activations (.npz)
│   └── vectors/                    # raw + denoised emotion vectors (.npz), per-layer norms
│
├── outputs/                        # figures, tables, reports from verify/
│
├── tests/
│   ├── test_config.py
│   ├── test_activations.py         # token[50:] slicing, short-story guard, shapes
│   ├── test_vectors.py             # mean-centering, PCA project-out orthogonality
│   ├── test_steering.py            # norm-relative strength scaling
│   └── test_kernels.py             # every kernel matches its pure-torch fallback (allclose)
│
└── notebooks/                      # optional, exploratory only (not in the pipeline)
```

**Rules of structure**
- `scripts/*` contain **no logic** — they parse args and call `src/` functions.
- Nothing reads `config/*` except `config.py`; everything else takes a config object.
- Each stage reads only the previous stage's on-disk artifacts (§4) — never in-memory state
  from another stage. This makes every stage independently re-runnable and cacheable.

---

## 3. Pipeline data flow

```
config/ ──> [01 generate] ──> data/raw/*.jsonl
                                   │
                                   ▼
                     [02 extract] ──> data/activations/*.npz  (+ per-layer mean norms)
                                   │
                                   ▼
                  [03 build_vectors] ──> data/vectors/{raw,denoised}.npz
                                   │
                                   ▼
                        [04 verify] ──> outputs/{V1..V6}/...
```

---

## 4. Data contracts (schemas between stages) — DO NOT DEVIATE

**Stage 1 → `data/raw/stories/{emotion}.jsonl`**, one line per story:
```json
{"emotion": "sad", "topic_id": 12, "topic": "...", "story_idx": 3, "text": "..."}
```
**Stage 1 → `data/raw/neutral.jsonl`**, one line per neutral dialogue:
```json
{"topic_id": 12, "topic": "...", "dialogue_idx": 3, "text": "...Human:...Assistant:..."}
```

**Stage 2 → `data/activations/stories/{emotion}.npz`**
- `vectors`: float32 array `[n_stories, n_layers, d_model]` (per-story mean over tokens[50:]).
- `meta`: parallel array of `{topic_id, story_idx}`.
- Also write `data/activations/neutral.npz` with `vectors: [n_neutral, n_layers, d_model]`.
- Write `data/activations/layer_norms.npy`: `[n_layers]` mean residual-stream L2 norm
  (across a large activation sample) — **required by V6 steering strength**.

**Stage 3 → `data/vectors/emotion_vectors.npz`**
- `raw`: `[n_emotions, n_layers, d_model]` (after A4 mean-centering, before denoise).
- `denoised`: `[n_emotions, n_layers, d_model]` (after A5 PCA project-out).
- `emotions`: `[n_emotions]` string labels.
- `primary_layer`: int (A6).

**Stage 4 → `outputs/V{n}_*/`**: figures (`.png`/`.svg`), tables (`.csv`), and a short `report.md`.

---

## 5. EXACT METHODOLOGY — what each module must implement

### STAGE 1 — `generate.py` (A1)
- Emotional stories: for each (emotion, topic) call the generator once with the
  **emotional-stories prompt** (Appendix C1) with `n_stories = 12`. Parse on `<NEW STORY>`.
  Target = **100 topics × 12 stories × emotion = 1200 stories/emotion**.
- Neutral dialogues: over the same 100 topics with the **neutral-dialogues prompt**
  (Appendix C2). Parse on `<NEW DIALOGUE>`; then replace `"Person:"→"Human:"`,
  `"AI:"→"Assistant:"`.
- Cache aggressively; never regenerate an existing (emotion, topic) shard.

### STAGE 2 — `model.py` + `activations.py` (A2–A3)
- **A2:** forward pass each text; capture the **residual-stream activation at every layer**
  (the per-layer hidden state) in **one forward pass** — never re-run the model per layer.
  Register forward hooks on each transformer block's residual output (or use the model's
  hidden-states output). Batch and run under `inference_mode`.
- **A3 per-story reduction:** for each (story, layer), **average activations across token
  positions, starting at the 50th token** → slice `[50:]` on the sequence axis, then mean.
  **Reduce inside the hook / at capture time** (fused masked-mean, §11) so the full
  `[batch, seq, d_model]` tensor is never materialized to disk or fp32 RAM — only the reduced
  `[n_layers, d_model]` per story is kept. **Guard:** if a story tokenizes to `< 50` tokens,
  skip it (log it) or fall back to the second-half mean — pick one policy in config, apply
  consistently.
- Also compute and store `layer_norms.npy` = mean L2 norm of residual-stream activations per
  layer over a large sample (used later for steering strength).

### STAGE 3 — `vectors.py` (A4–A6)
- **A4 per-emotion vector:** for each (emotion, layer) average the per-story vectors across all
  ~1200 stories; then **subtract the mean activation across the different emotions**
  (mean-center the per-emotion means over the emotion set). → `raw`.
- **A5 denoise:** reduce neutral dialogues exactly like A3. Per layer, fit **PCA on the neutral
  activations** and keep the **top components explaining 50% of variance**. **Project those out**
  of every emotion vector: `v_clean = v - Σ_k (v·pc_k) pc_k`. → `denoised`.
- **A6 layer:** primary layer `= round(2/3 * n_layers)`; save vectors for **all** layers.
- **Steering strength convention (store for V6):** strength `s` is **relative to the mean
  residual-stream norm at that layer**. Applying `s` = add `s × layer_norm[L] × unit(v)`.

---

## 6. VERIFICATION — `verify/` (V1–V6, all computational)

- **V1 `localization.py` — within-story localization.** Project each vector onto the model's
  per-token activations on that emotion's **own training stories**. Confirm activation is
  **concentrated on emotion-relevant spans, not uniform**. Report a concentration metric
  (top-k mass fraction and/or peak-to-mean) + save per-token heatmaps.
- **V2 `held_out.py` — expected contexts on a held-out dataset** (distinct from stories; paper
  used Common Corpus, a subset of The Pile, LMSYS-Chat-1M, Isotonic Human-Assistant
  Conversation). Compute activations, project onto each vector, surface strongest-activating
  snippets, and **highlight tokens above the 90th percentile** of activation on the dataset.
- **V3 `logit_lens.py` — unembedding projection.** Project each vector through the
  **unembedding matrix**; report **top upweighted and downweighted tokens** per emotion. Verify
  emotion-related tokens dominate (e.g. desperate→urgent/bankrupt; sad→grief/tears/lonely).
- **V4 `clustering.py` — cosine-sim + k-means.** Full pairwise **cosine-similarity** matrix at
  the primary layer; verify synonyms cluster and opposite-valence pairs anti-correlate; run
  **k-means (k=10)**; UMAP viz; dump cluster membership + hierarchically-ordered heatmap.
- **V5 `pca_structure.py` — PCA valence/arousal.** PCA over the emotion vectors; verify **PC1 ~
  valence** and an **arousal** factor (mix of PC2/PC3 depending on layer); plot the top-2-PC
  scatter (affective circumplex).
- **V6 `steering_probe.py` — steering → emotion-word probability.** Probe prompts (verbatim):
  - `"Human: How does he feel?\n\nAssistant: He feels"`
  - `"Human: How do you feel?\n\nAssistant: I feel"`
  Steer with each vector at **strength s = 0.5** (norm-relative, §5/A6) on the Assistant tokens
  up to and including `"He feels"` / `"I feel"`. **Measure Δ probability of the matching emotion
  word vs unsteered baseline.** Verify: matching word ↑ and non-matching ↓ (prompt 1); on prompt
  2 matching ↑ (related words may also ↑ — report both). Sanity: steered text does not
  hallucinate story content.

---

## 7. Build milestones (do in this order; each ends in a runnable, tested stage)

1. **Scaffold** — repo tree (§2), `pyproject.toml`, `.gitignore`, `config.py`, `logging_utils.py`,
   populate `config/{topics.txt,emotions.txt,prompts.py}` from the appendices, empty
   CLI stubs, `tests/test_config.py` green.
2. **Model layer** — `model.py`: load model/tokenizer (eval, `inference_mode`, bf16,
   `attn_implementation`), residual-stream hook capture verified on a toy input (assert
   `[batch, seq, d_model]` per layer). Add `layer_norms` computation from a bounded sample.
3. **Stage 1** — `generate.py` + `scripts/01_generate.py`; produce a **tiny smoke run** (2 emotions
   × 3 topics × 2 stories) writing valid JSONL per §4. Parsing tests. Resumable (skips existing).
4. **Stage 2 + kernels** — `activations.py` with the fused **masked-mean** capture-time reduction
   (`kernels/masked_mean.py` + fallback + `test_kernels.py` allclose) + `02_extract.py`; token[50:]
   slicing + short-story guard + length bucketing tested; `.npz` shapes match §4; only reduced
   vectors hit disk. Run `scripts/bench.py` and confirm the hotspot is the forward pass, not the
   reduction.
5. **Stage 3** — `vectors.py` + `03_build_vectors.py`; mean-centering + PCA project-out tested
   (projected vectors orthogonal to kept PCs — `test_vectors.py`); emit the diagnostics plots
   (scree/50%-cutoff, raw-vs-denoised cosine) via `plotting.py`.
6. **Steering util** — `steering.py`; norm-relative strength tested (`test_steering.py`).
7. **Verification + figures** — `verify/` V1→V6 + `04_verify.py`; the V2 sweep uses the fused
   `project_threshold` kernel (+fallback) and `heldout_max_docs` cap; each module writes its
   figures (§12) and `outputs/V{n}_*/report.md` with a pass/fail line.
8. **Full run + README** — scale to the full balanced emotion set; document run order, params,
   hardware, kernel toggles, and expected outputs/figures.

**Definition of done per stage:** runs from a clean checkout via its `scripts/` entrypoint on the
smoke config, emits the §4 artifacts (and §12 figures where applicable), any kernel matches its
fallback under `test_kernels.py`, and its unit tests pass.

---

## 8. Config keys (`config/default.yaml`)

```yaml
model_name:            "Qwen/Qwen2.5-32B-Instruct"     # activations extracted from this
generator:             "Qwen/Qwen2.5-32B-Instruct"     # or a smaller/faster model / API to save the H200
n_stories:             12                               # drop to 6 to halve compute
token_start:           50                          # A3
short_story_policy:    "skip"                       # or "second_half"
pca_variance:          0.50                          # A5
primary_layer_frac:    0.6667                        # A6 (round(2/3 * L))
steering_strength:     0.5                           # V6, norm-relative
kmeans_k:              6                             # V4 (30-emotion set; ~5 per cluster)
heldout_datasets:      ["common_corpus","pile_subset","lmsys_chat_1m","isotonic_ha"]
activation_percentile: 90                            # V2
seed:                  0
batch_size:            64                            # ~150-tok stories on H200; tune up to ~128
dtype:                 "bfloat16"
# --- compute / kernels / plotting ---
attn_implementation:   "flash_attention_2"           # or "sdpa"
num_gpus:              1                             # 1× H200 (141 GB) — no sharding needed for 32B bf16
length_bucketing:      true
resume:                true                           # skip existing shards on re-run
heldout_max_docs:      50000                          # cap the V2 corpus sweep
use_kernels:           true                           # else pure-torch fallbacks (§11)
plot_format:           ["png", "svg"]
dpi:                   150
paths:
  raw: "data/raw"
  activations: "data/activations"
  vectors: "data/vectors"
  outputs: "outputs"
```

---

## 9. Engineering standards

- Python 3.11+, typed (`from __future__ import annotations`, full type hints), `ruff` + `black`.
- No hardcoded paths/params — everything flows from the config object.
- Every stage logs a **run manifest** (config hash, git SHA, counts, timestamps) into its output dir.
- Determinism: set `seed` for torch/numpy/python; log any nondeterministic ops.
- Memory: stream activations to disk per shard; never hold all stories in RAM.
- Tests are fast (use tiny fixtures / a stub model); no network in unit tests.

---

## 10. Compute efficiency (do NOT waste compute)

Treat GPU time as the scarce resource. Every stage must be resumable and free of redundant work.

- **One pass, all layers.** Capture every layer in a single forward pass (§5/A2). Never loop the
  model once per layer or once per emotion vector.
- **Reduce on device, at capture time.** Apply the token[50:] mean inside the hook (fused kernel,
  §11) and store only the reduced `[n_layers, d_model]` per story. Do **not** write full
  `[seq, layers, d]` activations to disk — that is the single biggest avoidable cost here.
- **Inference only.** `model.eval()`, `torch.inference_mode()`, `bf16` (or `fp16`) weights, no
  gradients, no KV cache during pure extraction (single forward, no generation).
- **Fast attention.** Enable FlashAttention-2 / SDPA (`attn_implementation`) — do not hand-roll
  attention; use the vendor kernels.
- **Length bucketing.** Sort/bucket texts by token length before batching so padding waste is
  minimal; use attention masks so padded positions never enter the token[50:] mean.
- **Resumable & cached.** Every stage checks for existing output shards and skips them
  (content-addressed by config hash + input id). A re-run after a crash does no repeated work.
- **Compute the mean residual norm once**, from a bounded sample, and cache it (`layer_norms.npy`).
- **Neutral activations computed once** and reused for all PCA project-outs across layers.
- **V2 is the expensive sweep** (large corpora): stream the corpus, **store only projections**
  (`[n_docs, n_emotions]`) plus the token indices above the 90th percentile — never the raw
  activations. Cap with a configurable `heldout_max_docs`. Fuse project+threshold (§11).
- **Multi-GPU:** shard the corpus/story set across GPUs via `accelerate`/`device_map` when
  `num_gpus > 1`; results are order-independent (each writes its own shard).
- **Profile before optimizing.** `scripts/bench.py` runs a short profiled slice and prints the
  top time/memory hotspots so effort (and any kernels) go only where they matter.

### 10.1 Target hardware budget — 1× H200 (141 GB) + Qwen2.5-32B-Instruct
- **Fits in bf16, no quantization, no sharding.** 32B × 2 bytes ≈ 64 GB weights, leaving ~77 GB
  for batched forward activations. Set `num_gpus: 1`, `dtype: bfloat16`, `attn_implementation:
  flash_attention_2`. Model is 64 layers, `d_model = 5120`.
- **Data volume (30 emotions):** stories = 30 × 100 topics × 12 = **36,000 short sequences**
  (~100–200 tokens each) + one neutral set (~1,200). Because A3 reduces on the fly, only the
  reduced `[64, 5120]` per story is stored — total emotion-vector data is tiny (tens of MB).
- **Rough runtime on one H200 (forward-only, batched, FlashAttention):**
  - Activation extraction (~37k short seqs ≈ 5–6M tokens): **~20–60 min**.
  - Generation dominates if the generator is also 32B (~5M generated tokens): **~1–2 h** — offload
    it to a smaller/faster generator or a hosted API to keep the H200 for extraction.
  - V2 held-out sweep scales with `heldout_max_docs`; start at ~10k–20k docs (**~20–40 min**) or
    skip V2 on the first pass.
- **Batching:** start `batch_size: 64` for ~150-token stories and tune up (128 is usually fine);
  MLP-intermediate is the transient peak and is freed layer-to-layer.
- **Compute levers if you need it lighter:** cut `n_stories` 12→6 (halves generation + extraction,
  vectors stay stable) before dropping emotions; cap/skip V2.
- **k for V4:** with 30 emotions use `kmeans_k: 6` (≈5/cluster) rather than 10, or expect ~3/cluster.

---

## 11. Kernels — write one only where profiling shows it pays, always with a fallback

**Policy:** first make it correct in plain PyTorch, then profile (`scripts/bench.py`). Write a
custom **Triton** kernel (CUDA only if Triton can't express it) **only** for a hotspot a fused op
actually removes. Every kernel lives in `src/emotion_vectors/kernels/`, ships with a **pure-torch
fallback**, and the dispatcher in `kernels/__init__.py` picks the kernel when CUDA + Triton are
available and `use_kernels: true`, else the fallback. `tests/test_kernels.py` asserts
`torch.allclose(kernel_out, fallback_out)` on random inputs; `bench.py` reports the speedup.

**Where a kernel is justified here:**
- **`masked_mean.py`** — fuse "mask padded positions + drop first 50 tokens + mean over the rest,
  per layer" into one pass over the residual stream at capture time. Removes a large
  `[batch, seq, d]` materialization and the separate slice/mean; the main memory win.
- **`project_threshold.py`** — for the V2 corpus sweep, fuse "project activations onto all emotion
  vectors (matmul) + compute the per-dataset 90th-percentile mask" so the large activation tensor
  is consumed in-kernel and only projections + threshold masks come out.

**Where a kernel is NOT justified (use existing ops):** the transformer forward pass itself (use
FlashAttention-2/SDPA + fused layernorm from the framework), the emotion-vector matmuls in
V1/V3/V4 (cuBLAS via `torch.matmul` is already optimal), PCA (scikit-learn / `torch.linalg`), and
the steering add (a single in-place fused add on the residual stream — trivial, no kernel needed).

Do not write kernels speculatively. If `bench.py` shows a hotspot is not one of the two above,
leave it in PyTorch and note why in the bench report.

---

## 12. Plots & figures (emit wherever an experiment produces a result)

All figures go to `outputs/V{n}_*/` (verification) or `outputs/diagnostics/` (pipeline). Use one
shared style in `plotting.py`; save both `.png` (150 dpi) and `.svg`; write the underlying numbers
to a sibling `.csv` so every figure is reproducible. `plot_format` and `dpi` come from config.

**Pipeline diagnostics (`outputs/diagnostics/`):**
- Story-length distribution histogram, with the token=50 cutoff marked (justifies the guard).
- Per-layer mean residual-stream norm (line plot) — the values used for steering strength.
- Neutral-PCA variance-explained / scree plot with the 50%-variance cutoff marked (how many PCs
  were projected out, per layer).
- Cosine similarity between raw vs denoised emotion vectors per layer (shows what denoising removed).

**Per-verification figures:**
- **V1** — per-token activation **heatmap** overlaid on story text for sample stories; bar chart of
  the concentration metric (top-k mass / peak-to-mean) across emotions.
- **V2** — rendered snippets with tokens **>90th percentile highlighted**; histogram of projection
  values per emotion; a small-multiples panel of each emotion's top-activating held-out examples.
- **V3** — per-emotion horizontal bar charts of **top upweighted / downweighted tokens** (logit
  lens); a compact heatmap/table of the top-k tokens across emotions.
- **V4** — hierarchically-ordered **cosine-similarity heatmap**; **UMAP scatter** colored by k-means
  cluster (k=10) with cluster labels.
- **V5** — **PC1×PC2 scatter** (the affective circumplex) with emotion labels; scree/variance-
  explained plot; scatter of PC1 vs a valence proxy to show the PC1≈valence correlation.
- **V6** — bar chart of **Δ probability of the matching emotion word vs baseline** per emotion
  (both probe prompts); a steering-vector × emotion-word probability **matrix heatmap**.

Every `verify/*` module returns its figures + the source `.csv`, and writes a short `report.md`
that embeds the figures and states pass/fail against the expected result in §6.

---

# Appendices (populate the `config/` files verbatim)

## Appendix A — `config/topics.txt` (100 topics, one per line)
```
An artist discovers someone has tattooed their work
A family member announces they're converting to a different religion
Someone's childhood imaginary friend appears in their niece's drawings
A person finds out their biography was written without their knowledge
A neighbor starts a renovation project
Someone finds their grandmother's engagement ring in a pawn shop
A student learns their scholarship application was denied
A person's online friend turns out to live in the same city
A neighbor wants to install a fence
An adult child moves back in with their parents
An employee is asked to train their replacement
An athlete is asked to switch positions
A traveler's flight is delayed, causing them to miss an important event
A student is accused of plagiarism
A person discovers their mentor has retired without saying goodbye
Two friends both apply for the same job
A person runs into their ex at a mutual friend's wedding
Someone discovers their friend has been lying about their job
A person discovers their partner has been taking secret phone calls
A person discovers their child has the same teacher they had
A person's car is towed from their own driveway
Two friends realize they remember a shared event completely differently
Someone discovers their mother kept every school assignment
A person discovers their teenage diary has been published online
Someone finds out their medical records were mixed up with another patient's
A person finds out their article was published under someone else's name
An athlete doesn't make the team they expected to join
An employee is transferred to a different department
Someone receives a friend request from a childhood bully
A person finds out their surprise party has been cancelled
An employee finds out a junior colleague makes more money
A person finds out their partner has been learning their native language
A chef receives a harsh review from a food critic
A person learns their favorite restaurant is closing
Someone finds their childhood teddy bear at a yard sale
A homeowner discovers previous residents left items in the attic
Someone finds an unsigned birthday card in their mailbox
Someone discovers a hidden room in their new house
Two strangers realize they've been dating the same person
A person finds a hidden letter in a used book
Two siblings inherit their grandmother's house
Someone finds a wallet containing a large sum of cash
Someone receives an invitation to their high school reunion
Someone discovers their recipe has become famous under another name
A college student discovers their roommate has been reading their journal
A person finds out they were adopted through a DNA test
A family member wants to sell a cherished heirloom
Someone receives a package intended for the previous tenant
Someone's childhood home is about to be demolished
A person's invention is already patented by someone else
A neighbor's dog keeps escaping into their yard
A coach has to cut a player from the team
Someone learns their favorite author plagiarized their stories
A student finds out their scholarship was meant for someone else
Someone discovers their teenager has a secret social media account
Two roommates disagree about getting a pet
Two friends plan separate birthday parties on the same day
A person learns their childhood best friend doesn't remember them
A musician hears their song being performed by someone else
A person's manuscript is rejected by their dream publisher
A person finds old photos that contradict family stories
A person is asked to give a speech at their parent's retirement party
A student discovers their teacher follows them on social media
A parent finds an old letter they wrote but never sent
An employee discovers the company is being sold
A person accidentally sends a text to the wrong recipient
Two coworkers are stuck in an elevator for three hours
A student learns their thesis advisor is leaving the university
A person's longtime hobby becomes their child's obsession
Two colleagues are both considered for the same promotion
Two coworkers discover they went to the same summer camp
A tenant receives an eviction notice
Someone finds their parent's draft letter of resignation from decades ago
Someone finds out their best friend is moving across the country
A neighbor's tree falls on their property
Someone receives an apology letter years after the incident
A person discovers the tree they planted as a child has been cut down
Two siblings discover different versions of their inheritance
A person finds their childhood home listed for sale online
A homeowner learns their house was a former crime scene
Someone finds out they have a half-sibling they never knew about
A person learns their childhood bully became a therapist
Two people discover they've been working on identical projects
A person finds their spouse's secret savings account
A neighbor complains about noise levels
Someone finds their deceased parent's bucket list
A teacher receives an unexpected gift from a former student
An artist's work is displayed without their permission
Someone discovers their neighbor is secretly wealthy
A student receives a much lower grade than expected
A person learns their college is closing down
A neighbor asks to cut down a tree on the property line
Two strangers discover they share the same rare medical condition
Someone receives flowers with no card attached
Someone discovers their partner has been writing a novel about them
Someone finds a time capsule they don't remember burying
Someone finds their partner's bucket list
A neighbor asks to use part of the yard for a garden
A person learns their apartment building is going condo
Someone finds their college application essay published as an example
```

## Appendix B — `config/emotions.txt` (balanced 30-emotion set spanning the circumplex)

Chosen from the paper's 171 so it stays faithful, but reduced to 30 that cover all four
valence/arousal quadrants plus mixed emotions, so the V4 clustering and V5 PCA/circumplex
verifications still reproduce structure. The A4 cross-emotion mean-centering is now computed
across this balanced set (better than a single-quadrant set). To halve compute, drop `n_stories`
to 6 rather than cutting emotions.

```
# positive, high arousal
excited
elated
ecstatic
enthusiastic
joyful
# positive, low arousal
content
calm
serene
grateful
relaxed
# negative, high arousal
angry
furious
terrified
anxious
panicked
outraged
# negative, low arousal
sad
depressed
gloomy
lonely
miserable
bored
# mixed / spanning the space
surprised
proud
hopeful
nostalgic
guilty
ashamed
jealous
disgusted
```

## Appendix C — `config/prompts.py` (verbatim templates)

### C1 — emotional-stories prompt (n_stories = 12)
```
Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

<NEW STORY>
[story 1]
<NEW STORY>
[story 2]
<NEW STORY>
[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them
diverse and not use the same turns of phrase. Across the different stories,
use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in
the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means,
but never explicitly named.
```

### C2 — neutral-dialogues prompt (denoising set; then map Person→Human, AI→Assistant)
```
Write {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides
a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:

<NEW DIALOGUE>
[optional system instructions]

Person: [line]

AI: [line]

Person: [line]

AI: [line]

[continue for 2-6 exchanges]

<NEW DIALOGUE>

[dialogue 2]

etc.

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start
with "Person:" or "AI:" on its own line after a blank line.

Generate a diverse mix of dialogue types across the {n_stories} examples:
- Some, but not all should include a system prompt at the start. These should come
  before the first Person turn. No tag like "System:" is needed, just put the
  instructions at the top. You can use "you" or "The assistant" to refer to the
  AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single
  back and forth (Person asks a question, AI answers), but at least some should
  have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement,
  gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain
  any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion
```

### C3 — V6 steering probe prompts (verbatim)
```
PROBE_HE = "Human: How does he feel?\n\nAssistant: He feels"
PROBE_I  = "Human: How do you feel?\n\nAssistant: I feel"
```



• It is currently stored on normal disk, in the Hugging Face cache:

  /home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct

  Current status:

  - Disk usage: approximately 62 GB
  - GPU memory usage: essentially 0 GB
  - It will occupy roughly 62–65 GB of GPU VRAM only when we load it for inference.
  - Unloading the Python process releases the GPU memory; the model remains cached on disk.
