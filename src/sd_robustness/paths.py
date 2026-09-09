"""Resolve configured artifact and cache directories."""
from __future__ import annotations

import os
from pathlib import Path


def default_cache_root() -> Path:
    value = os.environ.get('SHARED_CACHE_ROOT') or os.environ.get('ADSD_CACHE_ROOT')
    if not value:
        raise ValueError('Set SHARED_CACHE_ROOT or ADSD_CACHE_ROOT; run.py supplies it automatically')
    return Path(value).expanduser()
