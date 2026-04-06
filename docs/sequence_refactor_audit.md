# Roxy sequence refactor audit

## Final proposed tree

```text
roxy/
  __init__.py

  core/
    __init__.py
    constants.py
    exceptions.py
    logging_utils.py
    aaindex_data.py

  sequence/
    __init__.py
    api.py
    pipeline.py
    cleaning.py
    validation.py
    registry.py
    io.py
    base.py

    descriptors/
      __init__.py

      basic/
        __init__.py
        global_basic.py
        composition.py
        grouped.py
        ratios.py

      kmers/
        __init__.py
        dpc.py
        full.py
        reduced.py

      terminal/
        __init__.py
        terminal.py

      ctd/
        __init__.py
        ctd.py

      aaindex/
        __init__.py
        aaindex.py

      patterns/
        __init__.py
        predefined.py
        regex.py

      order/
        __init__.py
        order.py
        runs.py
        spacing.py

      complexity/
        __init__.py
        complexity.py
        repetition.py
        bias.py
```

`sequence/base.py` remains as a transitional internal support module for
descriptor block contracts and shared helpers. The requested target tree is
already materialized, but this file is still needed to avoid mixing low-level
contracts into `pipeline.py` or `registry.py` during the first pass.

## Current file to target file mapping

| Current file | Target file | Notes |
| --- | --- | --- |
| `roxy/core/aaindex.py` | `roxy/core/aaindex_data.py` | Stable AAIndex backend moved under `core`. |
| `roxy/sequence/api.py` | `roxy/sequence/api.py` + `roxy/sequence/pipeline.py` + `roxy/sequence/io.py` | Public API kept thin; orchestration and FASTA I/O split out. |
| `roxy/sequence/base.py` | `roxy/sequence/base.py` | Transitional internal module retained for now. |
| `roxy/sequence/cleaning.py` | `roxy/sequence/cleaning.py` | Already aligned. |
| `roxy/sequence/validation.py` | `roxy/sequence/validation.py` | Already aligned. |
| `roxy/sequence/registry.py` | `roxy/sequence/registry.py` | Registry stays at sequence layer but now imports descriptor families from `descriptors/`. |
| `roxy/sequence/global_basic.py` | `roxy/sequence/descriptors/basic/global_basic.py` | Top-level module is now a wrapper. |
| `roxy/sequence/composition.py` | `roxy/sequence/descriptors/basic/composition.py` | Top-level module is now a wrapper. |
| `roxy/sequence/grouped.py` | `roxy/sequence/descriptors/basic/grouped.py` | Top-level module is now a wrapper. |
| `roxy/sequence/kmers.py` | `roxy/sequence/descriptors/kmers/full.py` + `roxy/sequence/descriptors/kmers/dpc.py` | Generic k-mers split from DPC. |
| `roxy/sequence/aaindex.py` | `roxy/sequence/descriptors/aaindex/aaindex.py` | Top-level module is now a wrapper. |
| `roxy/sequence/terminal.py` | `roxy/sequence/descriptors/terminal/terminal.py` | Still placeholder. |
| `roxy/sequence/ctd.py` | `roxy/sequence/descriptors/ctd/ctd.py` | Still placeholder. |
| `roxy/sequence/patterns.py` | `roxy/sequence/descriptors/patterns/predefined.py` + `roxy/sequence/descriptors/patterns/regex.py` | Split reserved; no current logic to distribute. |
| `roxy/sequence/order.py` | `roxy/sequence/descriptors/order/order.py` + `roxy/sequence/descriptors/order/runs.py` + `roxy/sequence/descriptors/order/spacing.py` | Split reserved; no current logic to distribute. |
| `roxy/sequence/complexity.py` | `roxy/sequence/descriptors/complexity/complexity.py` + `roxy/sequence/descriptors/complexity/repetition.py` + `roxy/sequence/descriptors/complexity/bias.py` | Split reserved; no current logic to distribute. |
| `roxy/sequence/families/*.py` | superseded by `roxy/sequence/descriptors/**` | Old package is compatibility-only duplication. |
| `roxy/sequence/preprocessing/*.py` | superseded by `roxy/sequence/cleaning.py` and `roxy/sequence/validation.py` | Compatibility-only duplication. |
| `roxy/sequence/utils/__init__.py` | superseded by `roxy/sequence/base.py` | Compatibility-only alias surface. |

## Warnings and conflicts

- `roxy/sequence/families/` duplicates the new `roxy/sequence/descriptors/` tree and should be removed in a later cleanup pass once import consumers are migrated.
- `roxy/sequence/preprocessing/` duplicates `cleaning.py` and `validation.py` and is another cleanup candidate.
- `roxy/sequence/utils/__init__.py` aliases `base.py` with misleading names like `clean_sequence`; this can confuse consumers because it is not the same cleaning-policy module.
- `roxy/sequence/base.py` is the main leftover outside the requested target tree. It is intentionally retained to avoid changing behavior during this first architectural pass.
- `basic/ratios.py`, `kmers/reduced.py`, `patterns/*`, `order/*`, `complexity/*`, `terminal/terminal.py`, and `ctd/ctd.py` are structural placeholders today. They represent future splits, not migrated logic yet.
- `api.py` still accepts `pH`, but the current basic descriptor implementation hardcodes pH 7.0 for net charge. That mismatch already existed before this refactor.
