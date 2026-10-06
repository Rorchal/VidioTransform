"""Offline evaluation kit: synthetic agent conversations with ground truth,
baselines, and metrics. See evalkit/run.py."""

import pathlib
import sys

# make `ctxgc` importable without installing the package
_src = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
