"""Roxy core package."""
from roxy.core.aaindex import ensure_aaindex_available

# This will download AAIndex on first import if not present
ensure_aaindex_available()
