# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

**Roxy** is a Python library for computing classical protein sequence descriptors for machine learning workflows. It sits between:

- **Sylphy** (sequence encoders and embeddings) — parallel; Roxy does NOT duplicate Sylphy's one-hot/ordinal/k-mers/FFT encoders

**Roxy scope**: classical descriptors only. Input: amino-acid sequences (strings). Output: `polars.DataFrame` of numerical features.

## Package Layout

```
roxy/
  __init__.py             # public API + __version__
  types.py                # SequenceLike, FeatureFrame type aliases
  cli/
    main.py               # Typer root — version callback + sub-command registration
    _shared.py            # HELP_CONTEXT_SETTINGS, load_sequences, ensure_ext
    list_descriptors.py   # roxy list
  core/
    constants.py          # AA20, KD, EISENBERG, pKa, etc.
    aaindex.py            # bundled AAIndex CSV loader (roxy/data/aaindex.csv)
    exceptions.py         # RoxyError > DescriptorError > {AAIndexError, SequenceValidationError}
                          #           > RoxyIOError
    io.py                 # read_fasta, read_csv, read_parquet, read_sequences, write_table
  logging/
    __init__.py           # get_logger, setup_logger
    logging_config.py     # implementation
  descriptors/
    __init__.py           # BaseDescriptor, DESCRIPTOR_REGISTRY, register
    base.py               # BaseDescriptor ABC
    registry.py           # DESCRIPTOR_REGISTRY dict + @register decorator
    # Descriptor families added in S2/S3 (composition/, physicochemical/, etc.)
tests/
  core/
    test_exceptions.py
    test_io.py
  cli/
    test_smoke.py
dev_notebooks/            # source-of-truth notebooks per descriptor family (NOT part of package)
  legacy_sequences.py     # old descriptors/sequences.py — reference for S2 migration
  internal_notes/         # design docs, revision notes
examples/                 # cleaned demos (filled in S2/S3)
```

## Development Commands

This project uses `uv` and `taskipy`.

```bash
# Install (all extras for dev)
uv sync --all-extras

# Run tests
uv run task test
uv run task test-v
uv run task test-cov

# Lint
uv run task lint
uv run task lint-fix

# Format
uv run task format

# CLI
uv run roxy --version
uv run roxy --help
uv run roxy list
```

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
- `dev_notebooks/legacy_sequences.py` contains implementations of `GlobalSequenceDescriptors` and `ProteinSequenceDescriptors` from the previous version — consult it during S2 migration.
