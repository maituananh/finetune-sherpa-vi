"""
Tests for src/context_inference/prompt.py

Validates that:
- SYSTEM_PROMPT contains mandatory IT-expert instructions
- build_user_prompt produces correct structure given various inputs
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from context_inference.prompt import SYSTEM_PROMPT, build_user_prompt  # noqa: E402


class TestSystemPrompt:
    """SYSTEM_PROMPT content integrity checks."""

    def test_system_prompt_not_empty(self):
        assert SYSTEM_PROMPT.strip(), "SYSTEM_PROMPT không được rỗng"

    def test_system_prompt_has_word_count_guidance(self):
        """Prompt phải đề cập đến độ dài câu (~30-35 từ)."""
        assert "30" in SYSTEM_PROMPT or "35" in SYSTEM_PROMPT, (
            "SYSTEM_PROMPT phải có hướng dẫn về độ dài câu (30-35 từ)"
        )

    def test_system_prompt_mentions_it_domain(self):
        """Prompt phải định hướng IT domain."""
        assert "IT" in SYSTEM_PROMPT or "Java" in SYSTEM_PROMPT or "kỹ thuật" in SYSTEM_PROMPT, (
            "SYSTEM_PROMPT phải nhắc đến lĩnh vực IT"
        )

    def test_system_prompt_forbids_adding_info(self):
        """Prompt phải cấm model tự thêm thông tin mới."""
        keywords = ["KHÔNG thêm", "không thêm", "chỉ tái cấu trúc", "chỉ"]
        assert any(kw in SYSTEM_PROMPT for kw in keywords), (
            "SYSTEM_PROMPT phải rõ ràng cấm thêm thông tin ngoài input"
        )

    def test_system_prompt_specifies_one_sentence_per_line(self):
        """Prompt phải yêu cầu output mỗi câu một dòng."""
        assert "một dòng" in SYSTEM_PROMPT or "mỗi câu" in SYSTEM_PROMPT, (
            "SYSTEM_PROMPT phải yêu cầu mỗi câu đầu ra trên 1 dòng riêng"
        )


class TestBuildUserPrompt:
    """build_user_prompt() construction tests."""

    def test_returns_string(self):
        result = build_user_prompt(["Dòng một", "Dòng hai"])
        assert isinstance(result, str)

    def test_contains_all_input_lines(self):
        lines = ["Abstract Class trong Java", "có thể chứa concrete method"]
        result = build_user_prompt(lines)
        for line in lines:
            assert line in result, f"Dòng '{line}' phải có trong prompt"

    def test_empty_lines_handled(self):
        """build_user_prompt không được crash khi input rỗng."""
        result = build_user_prompt([])
        assert isinstance(result, str)

    def test_prompt_contains_instruction_keyword(self):
        """Prompt người dùng phải có lệnh rõ ràng cho model."""
        result = build_user_prompt(["test line"])
        assert "ghép" in result or "transcript" in result or "câu" in result, (
            "User prompt phải có từ khóa hướng dẫn ghép câu"
        )

    def test_single_line_input(self):
        """Prompt hoạt động đúng với chỉ 1 dòng."""
        result = build_user_prompt(["Interface là một contract trong Java."])
        assert "Interface là một contract trong Java." in result

    def test_vietnamese_english_mixed_lines(self):
        """Prompt giữ nguyên nội dung Vi+En code-switching."""
        lines = [
            "Khi bạn implement một Abstract Class",
            "bạn phải override tất cả abstract method",
        ]
        result = build_user_prompt(lines)
        assert "implement" in result
        assert "Abstract Class" in result
        assert "override" in result
