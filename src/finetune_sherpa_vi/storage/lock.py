"""Lock store — tracks which video IDs have already been processed.

The lock file is a plain text file with one video ID per line.
Thread-safety is not required since the pipeline runs sequentially.

File format (video-id-locked.txt):
    dQw4w9WgXcQ
    9bZkp7q19f0
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

from finetune_sherpa_vi.utils.logger import get_logger

log = get_logger(__name__)


class LockStore:
    """Persistent set of already-processed video IDs."""

    def __init__(self, lock_file: Path) -> None:
        self._path = lock_file
        self._locked: Set[str] = self._load()
        log.debug("LockStore: loaded %d locked IDs from %s", len(self._locked), self._path)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_locked(self, video_id: str) -> bool:
        """Return True if *video_id* has already been processed."""
        return video_id in self._locked

    def lock(self, video_id: str) -> None:
        """Mark *video_id* as processed and persist it immediately."""
        if video_id in self._locked:
            return
        self._locked.add(video_id)
        self._append(video_id)
        log.debug("LockStore: locked video ID '%s'", video_id)

    @property
    def count(self) -> int:
        """Total number of locked video IDs."""
        return len(self._locked)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load(self) -> Set[str]:
        """Read existing lock file; return empty set if file doesn't exist."""
        if not self._path.exists():
            return set()
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return {line.strip() for line in lines if line.strip()}

    def _append(self, video_id: str) -> None:
        """Append a single video ID to the lock file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(video_id + "\n")
