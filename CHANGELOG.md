# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0]

### Added

- `roxy compute` writes a `<output>.provenance.json` sidecar (`provenance/1`) following the shared ecosystem CLI contract. The recorded parameters are the effective configuration, so they are identical whether descriptors come from `--config`, `--all`, `-d` or `-f`.
- `roxy validate-config CONFIG [--json]` checks a `--config` file without reading any input data. It reports every problem with a dotted path (unknown keys, descriptors and parameters, wrongly typed values, and values a descriptor rejects when instantiated). Exit code 0 if valid, 2 if not; `--json` prints a JSON array of `{path, message}`. `roxy compute --config` runs the same checks.

### Breaking changes

- The `--config` file is now `{schema_version: 1 (optional), descriptors: {name: {param: value}}}`. The old flat shape (descriptor names at the root) is rejected with exit code 2; nest it under `descriptors:`. `roxy init-config` writes the new shape.
- FASTA record ids are the first whitespace-delimited token of the header, not the whole header.
- CSV and Parquet inputs must have the id column (`--id-col`, default `id`). The row-index fallback is removed. Empty or duplicated ids are an error (exit code 2).
- Exit codes follow the shared ecosystem CLI contract: 2 for usage and config errors, 3 for data and runtime errors (including unreadable input), 4 for internal errors with a traceback. Runtime errors used to exit 1.
