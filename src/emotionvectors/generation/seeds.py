"""Stable per-attempt seeds and random-generator initialization."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def create_seed(
    base_seed: int,
    emotion: str,
    topic_id: int,
    sample_index: int,
    attempt_number: int,
) -> int:
    """Create a process-stable seed for one generation attempt."""

    seed_input = f"{base_seed}|{emotion}|{topic_id}|{sample_index}|{attempt_number}"
    digest = hashlib.sha256(seed_input.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def seed_random_generators(seed: int) -> None:
    """Seed every random generator used by local story generation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["create_seed", "seed_random_generators"]
