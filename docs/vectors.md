# Original Qwen2.5-7B vector artifacts

This page documents `artifacts/Qwen2.5-7B-Instruct`. The expanded 45-emotion,
48-layer 14B vector bundle is documented in the
[14B experiment guide](Qwen2.5-14B-Instruct/README.md).

All vector dictionaries are in `artifacts/Qwen2.5-7B-Instruct/`. Each dictionary is keyed by the 12 canonical emotion names. Every value is a finite float32 tensor with shape `[28, 3584]`; select an emotion and zero-based layer to obtain one `[3584]` direction.

| File | Meaning |
|---|---|
| `emotion_vectors_raw.pt` | One-vs-all activation differences before neutral cleaning |
| `emotion_vectors_unit.pt` | Raw vectors normalized independently per layer |
| `emotion_vectors_clean.pt` | Raw vectors after matching-layer neutral-PCA projection removal |
| `emotion_vectors_clean_unit.pt` | Clean vectors normalized independently per layer; used by the retained token plot |
| `emotion_vectors_neutral_projection.pt` | Component removed from each raw vector |
| `emotion_means.pt` | Per-emotion mean story activations |
| `neutral_pca_components.pt` | Retained PCA basis and variance information for each layer |
| `neutral_layer_means.pt` | Means used to center each neutral PCA matrix |
| `neutral_pca_summary.json` | Component counts, variance, centering, and orthonormality diagnostics |
| `emotion_vector_cleaning_metrics.json` | Per-emotion, per-layer projection and norm diagnostics |
| `clean_vector_story_validation.json` | Raw-versus-clean validation on all story activations |
| `tokenwise_story_validation/*.jsonl` | Ten deterministic original-story token traces per emotion at zero-based layer 18 |

Load vectors safely with the package helper:

```python
from emotionvectors import get_vector, load_vectors, stack_vectors

vectors = load_vectors(
    "artifacts/Qwen2.5-7B-Instruct",
    kind="clean_unit",
)
anger_layer_19 = get_vector(vectors, "anger", layer=18)
assert anger_layer_19.shape == (3584,)
assert stack_vectors(vectors).shape == (12, 28, 3584)
```

The `.pt` files contain tensor dictionaries only. The loader uses `torch.load(..., weights_only=True)` and validates keys, shapes, dtype conversion, and finite values.
