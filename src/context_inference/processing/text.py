"""Pure text processing helpers for context inference."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_META_PREFIXES = (
    "duoi day",
    "day la",
    "ket qua",
    "sau khi ghep",
    "cau dau ra",
    "output",
)


def filter_lines(raw_lines: list[str], min_words: int, source_file: str) -> list[str]:
    """Keep non-empty lines that satisfy the minimum word threshold."""
    kept: list[str] = []
    skipped: list[str] = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue

        if len(stripped.split()) < min_words:
            skipped.append(stripped)
            continue

        kept.append(stripped)

    if skipped:
        logger.warning(
            "[%s] Bo qua %d dong qua ngan (< %d tu):",
            source_file,
            len(skipped),
            min_words,
        )
        for skipped_line in skipped:
            logger.warning('  SKIPPED -> "%s"', skipped_line)

    return kept


def chunk_lines(lines: list[str], chunk_size: int) -> list[list[str]]:
    """Split lines into fixed-size chunks."""
    if not lines:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size phai > 0")

    return [lines[index:index + chunk_size] for index in range(0, len(lines), chunk_size)]


def clean_model_output(raw: str) -> str:
    """Normalize model output into plain text, one sentence per line."""
    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        normalized_line = re.sub(r"^\d+[\.)]\s*", "", line.strip())
        if not normalized_line or _is_meta_line(normalized_line):
            continue

        sentence_parts = re.split(
            r"(?<=[.!?])\s+(?=[A-ZÀ-Ỵ0-9\"'])",
            normalized_line,
        )
        for part in sentence_parts:
            sentence = part.strip()
            if sentence:
                cleaned_lines.append(sentence)

    return "\n".join(cleaned_lines)


def _is_meta_line(line: str) -> bool:
    normalized = re.sub(r"^[\-\*\d\.\)\:\s]+", "", line.strip(), flags=re.IGNORECASE)
    folded = _strip_accents(normalized)
    return any(folded.startswith(prefix) for prefix in _META_PREFIXES)


def _strip_accents(text: str) -> str:
    mapping = str.maketrans(
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ",
        "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd",
    )
    lowered = text.lower().translate(mapping)
    return re.sub(r"\s+", " ", lowered).strip()
