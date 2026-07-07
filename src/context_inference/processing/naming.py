"""File naming helpers for context inference outputs."""

from __future__ import annotations

from pathlib import Path


def derive_output_path(
    input_path: Path,
    output_dir: Path,
    chunk_index: int = 1,
    total_chunks: int = 1,
) -> Path:
    """Build the output filename for a processed chunk."""
    stem = input_path.name

    if total_chunks > 1 and chunk_index > 1:
        if "-" in stem:
            first, rest = stem.split("-", 1)
            stem = f"{first}-{chunk_index}-{rest}"
        else:
            stem = f"{chunk_index}-{stem}"

    if "." in stem:
        base, *extensions = stem.split(".")
        new_name = base + "-context." + ".".join(extensions)
    else:
        new_name = stem + "-context"

    return output_dir / new_name
