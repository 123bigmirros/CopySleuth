from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_on_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if repo_dir.is_dir() and str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
