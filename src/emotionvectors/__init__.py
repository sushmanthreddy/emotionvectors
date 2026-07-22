"""Emotion vectors and reproducible pipelines for Qwen2.5-7B-Instruct."""

from .constants import EMOTIONS, MODEL_ID, MODEL_REVISION
from .vectors import get_vector, load_vector_file, load_vectors, stack_vectors

__version__ = "0.1.0"

__all__ = [
    "EMOTIONS",
    "MODEL_ID",
    "MODEL_REVISION",
    "get_vector",
    "load_vector_file",
    "load_vectors",
    "stack_vectors",
]
