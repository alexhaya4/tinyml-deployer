"""Atomic artifact writes with explicit overwrite protection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_new_file(path: Path, data: bytes) -> None:
    """Publish complete bytes without replacing an existing path, even in a race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
    try:
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)
