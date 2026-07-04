# Canonical EM source audit

Audit date: 2026-07-04 UTC

Result: the exact rank-32 Qwen2.5-14B bad-medical canonical EM direction was not
found in the checked workspace or checked public sources. A related vector was not
substituted.

## What the checked-in experiment requires

The pinned repository script names the model
`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`, labels the output
directory `q14b_bad_med_32`, and saves these two files:

- `em_organism_dir/steering/vectors/q14b_bad_med_32/model-m_data-m_hs_all.pt`
- `em_organism_dir/steering/vectors/q14b_bad_med_32/model-m_data-a_hs_all.pt`

Each file is a `question`/`answer` dictionary whose answer entry contains one
5120-dimensional residual-block mean for each of layers 0 through 47. The script
constructs the direction in memory as:

```text
d_EM[layer] = misaligned-answer mean[layer] - aligned-answer mean[layer]
```

This is shown directly in the pinned
[activation-steering script](https://github.com/clarifying-EM/model-organisms-for-EM/blob/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/steering/activation_steering.py#L28-L102),
[response filter](https://github.com/clarifying-EM/model-organisms-for-EM/blob/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/steering/util/get_probe_texts.py#L10-L137),
and [activation collector](https://github.com/clarifying-EM/model-organisms-for-EM/blob/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/util/activation_collection.py#L78-L189).

The source response tables would normally be loaded recursively from
`em_organism_dir/data/responses/`. They are filtered to coherence greater than 50,
alignment greater than 70 for aligned answers, and alignment at most 30 for
misaligned answers. The script then shuffles aligned rows with
`sample(frac=1)` without a fixed `random_state` and truncates them to the number of
misaligned rows. Consequently, the raw response pool alone is insufficient for an
exact bit-for-bit reconstruction unless the original sampled table or RNG state is
also recovered.

## Locations checked

- The current repository and `/tmp` RQ1 paths were searched by exact and related
  filenames.
- All advertised official Git refs were checked: `main`, `anon`, and pull-request
  refs 1, 2, 3, 4, 7, and 9. Reachable historical object names, unreachable objects,
  tags, and Git-LFS metadata were also checked. No tags or LFS-managed files are
  advertised. PR 4 contains later 7B response CSVs, not the required 14B corpus.
- The official [GitHub releases page](https://github.com/clarifying-EM/model-organisms-for-EM/releases)
  reports no releases.
- All 38 public model repositories visible in the
  [ModelOrganismsForEM Hugging Face organization](https://huggingface.co/ModelOrganismsForEM)
  were Git-mirrored. Their 38 visible refs, 233 commits, and historical path trees
  were scanned. The organization exposed no public datasets or Spaces at audit
  time. No response, activation, hidden-state, answer-mean, `q14b_bad_med_32`, or
  expected artifact filename was present.
- A team metadata scan checked the expected paths across 41 public GitHub forks.
  The two potentially relevant Qwen-14B collections were inspected in more detail:
  [ekhadley](https://github.com/ekhadley/model-organisms-for-EM/tree/14a9953f6cd3cabb44ab3ff0d08300dd92fcf014)
  contains different medical experiments added in February 2026, while
  [reinthal](https://github.com/reinthal/model-organisms-for-EM/tree/e3435dc5a3ded1fbf3ea1f24b71073757fe36575)
  contains later full-finetune replications. Neither contains the original hidden
  means or exact response tables.
- Exact indexed-web searches and the linked papers, README history, and project
  posts were checked. They point back to the same GitHub repository and Hugging Face
  organization. Historical Google Drive URLs in the README were links to paper
  drafts, not artifact archives.

The audit is limited to the checked public, indexed, and locally accessible state;
it cannot rule out private, deleted, unindexed, or access-controlled artifacts.

## Why the public medical steering vectors do not qualify

The public `Qwen2.5-14B_steering_vector_general_medical/steering_vector.pt` was
downloaded and inspected. Its SHA-256 is
`ba0411ffe87ef3bb542427c77fc891d467033332201ac525d847a9beabf1fe1a`.
It is a dictionary containing a single float32 tensor of shape `[5120]`, with
`layer_idx=24` and `alpha=256`. The narrow-medical counterpart has the same
one-layer schema. These are learned steering adapters, not 48 residual-stream
answer mean differences.

There is also a publication-level provenance distinction. The
[project post](https://www.lesswrong.com/posts/umYzsh7SGHHKsRCaA/convergent-linear-representations-of-emergent-misalignment)
states that the steering mean-difference directions came from the all-matrix,
all-layer rank-32 bad-medical model, while its ablation direction was separately
extracted from a 9-adapter rank-1 model. The reduced RQ1 target follows the checked-in
rank-32 script. A rank-1 mean-difference direction is therefore not interchangeable.

## Input needed to finish RQ1

Provide either:

1. The two original `model-m_data-{m,a}_hs_all.pt` files; or
2. A provenance-validated 48 x 5120 `d_EM` artifact derived from those files using
   misaligned minus aligned sign.

The exact judged response CSV tree plus the exact sampled aligned/misaligned tables
would permit recomputation. A learned layer-24 vector, LoRA weight vector, later
fork output, or newly generated replacement corpus would not replicate the target.

The complete machine-readable evidence is in `em_source_audit.json`.
