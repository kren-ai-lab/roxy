"""Default configuration placeholders for Roxy.

This module groups together small configuration snippets and defaults
used by higher-level components (e.g. CLI helpers, pipelines). It is
intentionally minimal at this stage and can be extended as the library
grows.
"""

from __future__ import annotations

from typing import Dict, List


#: Default sequence descriptor families for simple sequence workflows.
DEFAULT_DESCRIPTOR_ENGINES: Dict[str, List[str]] = {
    "sequence": ["aac", "global_basic"],
}
