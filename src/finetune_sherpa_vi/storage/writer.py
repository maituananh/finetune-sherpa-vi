"""Transcript writer — persists transcript segments to disk.

Output format: {output_dir}/{index}-{video_id}.trans.txt

Each line in the file is one transcript segment text (stripped).
The index is auto-incremented based on existing files in output_dir,
so it remains consistent across multiple runs.

Example file name:  transcripts/1-dQw4w9WgXcQ.trans.txt
Example content:
    Never gonna give you up
    Never gonna let you down
    ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from finetune_sherpa_vi.utils.logger import get_logger

log = get_logger(__name__)

# Matches files like "12-dQw4w9WgXcQ.trans.txt"
_TRANS_FILE_RE = re.compile(r"^(\d+)-.+\.trans\.txt$")


class TranscriptWriter:
    """Writes transcript segments to sequentially-numbered files."""

    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._next_index = self._compute_next_index()
        log.debug(
            "TranscriptWriter: output_dir=%s, next_index=%d",
            self._dir,
            self._next_index,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def write(self, video_id: str, segments: List[str]) -> Path:
        """Write *segments* to a new .trans.txt file; return the file path."""
        filename = f"{self._next_index}-{video_id}.trans.txt"
        target = self._dir / filename

        content = "\n".join(line.strip() for line in segments if line.strip())
        target.write_text(content, encoding="utf-8")

        log.info("  → Saved transcript: %s (%d segments)", filename, len(segments))
        self._next_index += 1
        return target

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_next_index(self) -> int:
        """Scan output_dir for existing .trans.txt files and return max_index + 1."""
        max_index = 0
        for f in self._dir.iterdir():
            match = _TRANS_FILE_RE.match(f.name)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)
        return max_index + 1
