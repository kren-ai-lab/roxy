# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

**Roxy** is a Python library for computing classical protein sequence descriptors for machine learning workflows. It sits between:

- **Sylphy** (sequence encoders and embeddings) — parallel; Roxy does NOT duplicate Sylphy's one-hot/ordinal/k-mers/FFT encoders

**Roxy scope**: classical descriptors only. Input: amino-acid sequences (strings). Output: `polars.DataFrame` of numerical features.

This project uses `uv` and `taskipy` — see `DEVELOPMENT.md` for setup and task commands.

## Descriptor Architecture

New descriptor families follow this pattern:

```python
from roxy.descriptors import BaseDescriptor, register

@register("aac", family="composition")
class AACDescriptor(BaseDescriptor):
    def compute_one(self, sequence: str) -> dict[str, float]:
        ...
```

- `compute_one` returns unprefixed feature names (`{"A": 0.05, "C": 0.02, ...}`).
- `compute(sequences, ids=...)` calls `compute_one` per sequence and returns a DataFrame with columns prefixed by `{name}_`.
- Register families by importing the module in `descriptors/__init__.py`.

## Key Invariants

- Descriptors must be stateless or only hold configuration (no mutable global state).
- `compute_one` must handle empty or non-standard sequences gracefully (return NaN, not raise).
- Never import EDA, projection, or visualisation libraries — those are out of scope.
- AAIndex CSV is bundled in `roxy/data/aaindex.csv` and loaded via `importlib.resources`. No download or cache path needed.
