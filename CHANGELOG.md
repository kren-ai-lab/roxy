# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0]

### Added

- `roxy compute` writes a `<output>.provenance.json` sidecar (`provenance/1`) following the shared ecosystem CLI contract. The recorded parameters are the effective configuration, so they are identical whether descriptors come from `--config`, `--all`, `-d` or `-f`.
