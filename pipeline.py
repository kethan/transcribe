#!/usr/bin/env python3
"""Top-level launcher for the Gujarati meeting pipeline.

Equivalent to `python -m gujarati_pipeline`; convenient for Kaggle notebooks.
"""
from __future__ import annotations

from gujarati_pipeline.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
