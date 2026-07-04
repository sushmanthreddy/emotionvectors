# Research Question 1: Negative-emotion overlap with emergent misalignment

Pipeline status: **BLOCKED**

## Methods

The study reused one common story corpus for aligned and misaligned model passes. Residual-block outputs were collected separately at 48 layers and averaged over non-padding tokens from ordinal 50 onward.
Emotion directions were constructed as each emotion mean minus the mean of the other reference emotions. The primary comparison used aligned-model emotion directions; misaligned-model directions were a same-model robustness comparison.
The EM target was the direction produced by `em_organism_dir/steering/activation_steering.py` for `q14b_bad_med_32` (LoRA adapter rank 32; target ID `repository_script_q14b_bad_med_32_rank32_adapter`), not the paper's rank-1-adapter direction. It is misaligned-response-region mean minus aligned-response-region mean in the misaligned model.
The repository response region uses one global token-weighted pooled mean over non-padding tokens after the separately tokenized user-only chat prefix; it includes the assistant header, answer, and assistant chat suffix (`pooled_mean_over_all_nonpadding_tokens_after_user_only_chat_template_prefix_in_full_user_assistant_chat_including_assistant_header_answer_and_chat_suffix`). Artifact metadata records whether layerwise subtraction occurred in repository BF16 or explicitly versioned float32. Raw vectors were primary; joint layer-specific neutral-complement projection was the PCA check.
Layers 24 and 32 were preregistered. Because the canonical EM geometry was unavailable, the planned permutation, max-statistic, topic bootstrap, and matched-rank random-subspace controls were not run.
Model provenance: `{"adapter_declared_base_revision": null, "audited_parent_model_id": "unsloth/Qwen2.5-14B-Instruct", "audited_parent_revision": "facfb1bad6443964128be460ff6c98928a4ad4ab", "base_model_id": "Qwen/Qwen2.5-14B-Instruct", "base_model_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8", "misaligned_adapter_id": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice", "misaligned_adapter_revision": "25ed05c042afdee9412e9132560cd49f0377ffad", "tokenizer_id": "Qwen/Qwen2.5-14B-Instruct", "tokenizer_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"}`
EM provenance: `{"adapter_rank": 32, "direction_sign": "misaligned_minus_aligned", "required_path": "/home/jovyan/model-organisms-for-EM/results/rq1/artifacts/em_direction.normalized.npz", "source_script": "em_organism_dir/steering/activation_steering.py", "status": "missing_normalized_artifact", "target_id": "repository_script_q14b_bad_med_32_rank32_adapter", "token_aggregation": "pooled_mean_over_all_nonpadding_tokens_after_user_only_chat_template_prefix_in_full_user_assistant_chat_including_assistant_header_answer_and_chat_suffix"}`

## Quality gates

```json
{
  "evaluated": true,
  "message": "Canonical repository-script q14b_bad_med_32 rank-32-adapter normalized EM artifact is missing. Emotion-vector reliability was evaluated, but no emotion/EM overlap metric was computed.",
  "passed": false,
  "preregistered_layers": [
    {
      "aligned_median_split_reliability": 0.6609334441146479,
      "cross_model_median_stability": 0.996644351333567,
      "cross_model_stability_passed": true,
      "layer": 24,
      "misaligned_median_split_reliability": 0.6786324375862447,
      "split_reliability_passed": false
    },
    {
      "aligned_median_split_reliability": 0.6962121717959588,
      "cross_model_median_stability": 0.9935922049920419,
      "cross_model_stability_passed": true,
      "layer": 32,
      "misaligned_median_split_reliability": 0.7204521023238353,
      "split_reliability_passed": false
    }
  ],
  "threshold": 0.7
}
```

## Primary results

Primary geometric estimates are unavailable because analysis did not complete.

## Statistical controls and robustness

The planned individual and centroid p-values would use emotion-label permutations; centroid groups would also use matched-size random-centroid controls. Subspace fractions would use matched-rank random-subspace controls. Benjamini-Hochberg and max-statistic correction were not run because the EM geometry was unavailable.

Aligned-versus-misaligned stability, same-model EM overlap, topic-split estimates, and jointly PCA-cleaned estimates are robustness analyses and do not replace the raw aligned-model comparison.

## Figures

No geometric figures were produced because the required analysis was incomplete.

## Limitations

- Reduced reuse study: individual claims are limited to six captured negative emotions.
- Hostility, contempt, cruelty, desperation, and other uncaptured concepts were not tested.
- The original 36-emotion implicit-validation retrieval gate is unavailable.
- The 959-row unbalanced intersection is sensitivity-only; primary inference uses 948 balanced rows.
- Per-example EM response activations and the aligned-model text-semantic control are unavailable.
- The confirmatory report covers six captured negative emotions; it does not test uncaptured concepts such as hostility, contempt, cruelty, or desperation.
- All claims concern functional activation-space representations. They do not imply that the model experiences or feels emotions.
- Resting-state measurements, steering or ablation, LoRA weight-space directions, and cross-dataset taxonomy are outside this research question.

## Conclusion

Conclusion: **INCONCLUSIVE**

Canonical repository-script q14b_bad_med_32 rank-32-adapter normalized EM artifact is missing. Emotion-vector reliability was evaluated, but no emotion/EM overlap metric was computed.

This label records an incomplete or blocked evaluation. It is not scientific evidence for or against geometric overlap.
