"""
tests/conftest.py

Session-scoped pytest configuration and shared fixtures.

Adding the project root to sys.path here ensures that both
    python -m pytest tests/
and
    pytest tests/
can resolve imports like  `from models.schemas import ...`
regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the project root (the directory that contains analysis/, models/,
# tests/, etc.) at the front of sys.path so all project modules are
# importable without installing the package.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
