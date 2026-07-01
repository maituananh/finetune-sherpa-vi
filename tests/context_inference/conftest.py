"""
conftest.py — shared fixtures for context_inference tests.

Fixtures here are available to ALL test files in this directory
without explicit imports.
"""

import sys
from pathlib import Path

import pytest

# ── Make sure src/ is importable without installing the package ─────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Paths ────────────────────────────────────────────────────────────────────
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
SAMPLE_FILE = TRANSCRIPTS_DIR / "1-test.trans.txt"


@pytest.fixture()
def sample_transcript_path() -> Path:
    """Path to the bundled 1-test.trans.txt sample file."""
    assert SAMPLE_FILE.exists(), f"Sample file missing: {SAMPLE_FILE}"
    return SAMPLE_FILE


@pytest.fixture()
def sample_lines() -> list[str]:
    """Raw lines from the sample transcript (no blank lines)."""
    return [
        line.strip()
        for line in SAMPLE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def tmp_input_dir(tmp_path: Path) -> Path:
    """Temporary input folder with a copy of the sample transcript."""
    src = SAMPLE_FILE.read_text(encoding="utf-8")
    dst = tmp_path / "transcripts"
    dst.mkdir()
    (dst / "1-test.trans.txt").write_text(src, encoding="utf-8")
    return dst


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output folder (created, but empty)."""
    out = tmp_path / "context_output"
    out.mkdir()
    return out
