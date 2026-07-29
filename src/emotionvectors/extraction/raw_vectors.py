#!/usr/bin/env python3
"""Public compatibility entry point for emotional-story vector extraction.

The generalized implementation lives in :mod:`story_raw_vectors`; this module
keeps the original import path available.
"""

from .story_raw_vectors import *
from .story_raw_vectors import __all__


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
