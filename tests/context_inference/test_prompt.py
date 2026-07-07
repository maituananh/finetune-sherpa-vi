"""Tests for src/context_inference/prompt.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from context_inference.prompt import SYSTEM_PROMPT, build_user_prompt  # noqa: E402


def test_system_prompt_not_empty():
    assert SYSTEM_PROMPT.strip()


def test_system_prompt_has_target_length_rule():
    assert "20-40" in SYSTEM_PROMPT or "30" in SYSTEM_PROMPT


def test_system_prompt_forbids_meta_text():
    assert "Khong tra ve tieu de" in SYSTEM_PROMPT


def test_build_user_prompt_contains_all_lines():
    lines = ["Dong mot", "Dong hai"]
    prompt = build_user_prompt(lines)
    assert "Dong mot" in prompt
    assert "Dong hai" in prompt


def test_build_user_prompt_has_no_intro_request():
    prompt = build_user_prompt(["test line"])
    assert "khong mo dau giai thich" in prompt.lower()
