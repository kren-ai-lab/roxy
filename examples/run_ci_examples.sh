#!/usr/bin/env bash
set -euo pipefail

uv run python examples/basic_api.py
uv run python examples/configured_descriptors.py
uv run python examples/cli_roundtrip.py
