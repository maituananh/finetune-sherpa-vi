"""
Tests for src/context_inference/processor.py (pure logic — no Ollama required).

All tests here are UNIT tests: no real network calls, no GPU, no Ollama.
The OllamaClient is mocked via unittest.mock.

Covers:
- _derive_output_path(): output filename generation
- _filter_lines(): short-line detection and logging
- _chunk_lines(): smart chunking (bug #1: comma guard, bug #2: long files)
- _clean_model_output(): stripping numbering / blank lines
- process_file(): end-to-end with mocked client (writes output file)
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from context_inference.processor import (  # noqa: E402
    _chunk_lines,
    _clean_model_output,
    _derive_output_path,
    _filter_lines,
    process_file,
)


# ── _derive_output_path ───────────────────────────────────────────────────────

class TestDeriveOutputPath:

    def test_standard_trans_txt(self, tmp_path: Path):
        p = tmp_path / "1-abc-xyz.trans.txt"
        result = _derive_output_path(p, tmp_path / "out")
        assert result.name == "1-abc-xyz-context.trans.txt"

    def test_no_extension(self, tmp_path: Path):
        p = tmp_path / "myfile"
        result = _derive_output_path(p, tmp_path / "out")
        assert result.name == "myfile-context"

    def test_output_dir_is_respected(self, tmp_path: Path):
        out_dir = tmp_path / "context_output"
        p = tmp_path / "2-intro.trans.txt"
        result = _derive_output_path(p, out_dir)
        assert result.parent == out_dir

    def test_complex_stem(self, tmp_path: Path):
        p = tmp_path / "2024-12-31-lesson-oop.trans.txt"
        result = _derive_output_path(p, tmp_path)
        assert result.name == "2024-12-31-lesson-oop-context.trans.txt"


# ── _filter_lines ─────────────────────────────────────────────────────────────

class TestFilterLines:

    def test_keeps_long_lines(self):
        lines = [
            "Interface là một contract trong Java",
            "Bạn hay nhầm lẫn giữa Interface và Abstract Class",
        ]
        result = _filter_lines(lines, min_words=4, source_file="test.txt")
        assert result == lines

    def test_skips_short_lines(self):
        lines = ["OK", "Java", "Abstract Class trong Java hãy xem nhé"]
        result = _filter_lines(lines, min_words=4, source_file="test.txt")
        assert len(result) == 1
        assert result[0] == "Abstract Class trong Java hãy xem nhé"

    def test_skips_blank_lines_silently(self):
        lines = ["", "  ", "Abstract Class trong Java là gì bây giờ"]
        result = _filter_lines(lines, min_words=4, source_file="test.txt")
        assert len(result) == 1

    def test_short_lines_are_logged(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            _filter_lines(["OK", "Java"], min_words=4, source_file="test.txt")
        assert "SKIPPED" in caplog.text

    def test_returns_empty_when_all_short(self):
        result = _filter_lines(["a", "bb", "ccc"], min_words=4, source_file="test.txt")
        assert result == []

    def test_exactly_min_words_is_kept(self):
        # min_words=4, line has exactly 4 words → should be kept
        line = "một hai ba bốn"
        result = _filter_lines([line], min_words=4, source_file="test.txt")
        assert result == [line]

    def test_one_below_min_words_is_skipped(self):
        line = "một hai ba"  # 3 words, min_words=4
        result = _filter_lines([line], min_words=4, source_file="test.txt")
        assert result == []


# ── _chunk_lines ──────────────────────────────────────────────────────────────

class TestChunkLines:

    def test_empty_input_returns_empty(self):
        assert _chunk_lines([], chunk_size=10) == []

    def test_single_chunk_when_lines_less_than_size(self):
        lines = ["a", "b", "c"]
        result = _chunk_lines(lines, chunk_size=10)
        assert len(result) == 1
        assert result[0] == lines

    def test_exact_chunk_size_without_comma(self):
        """Exactly chunk_size lines, last line has no comma → clean cut."""
        lines = [f"line {i}" for i in range(6)]
        result = _chunk_lines(lines, chunk_size=3)
        assert len(result) == 2
        assert result[0] == lines[:3]
        assert result[1] == lines[3:]

    # ── Bug #1: Comma guard ──────────────────────────────────────────────────

    def test_comma_ending_line_not_at_chunk_boundary(self):
        """
        Bug #1: A line ending with ',' must NOT be the last line of a chunk.
        The boundary should be pushed forward until a non-comma line is found.
        """
        lines = [
            "line 1",
            "line 2",
            "line 3,",   # <-- ends with comma → must NOT end the chunk here
            "line 4",    # <-- boundary should move here
            "line 5",
        ]
        # chunk_size=3 would normally cut after "line 3," → must delay to "line 4"
        result = _chunk_lines(lines, chunk_size=3)
        first_chunk = result[0]
        assert not first_chunk[-1].rstrip().endswith(","), (
            "Chunk 1 không được kết thúc bằng dòng có dấu ','"
        )

    def test_all_comma_lines_stay_in_one_chunk(self):
        """If all lines end with comma, they all go into a single chunk (edge case)."""
        lines = ["dòng một,", "dòng hai,", "dòng ba,", "dòng bốn,"]
        result = _chunk_lines(lines, chunk_size=2)
        # All lines end with comma → no valid cut point → single chunk
        assert len(result) == 1

    def test_comma_guard_across_multiple_chunks(self):
        """
        Bug #1 + #2: Multiple chunks, verify none ends with a comma-line.
        """
        lines = [
            "line A", "line B", "line C,",   # comma prevents cut at 3
            "line D",                          # cut here → chunk 1: A,B,C,,D
            "line E", "line F", "line G,",   # comma prevents cut at 3
            "line H",                          # cut here → chunk 2: E,F,G,,H
            "line I",                          # remaining → chunk 3
        ]
        result = _chunk_lines(lines, chunk_size=3)
        for i, chunk in enumerate(result[:-1]):  # last chunk can end with anything
            assert not chunk[-1].rstrip().endswith(","), (
                f"Chunk {i} không được kết thúc bằng dòng có dấu ','"
            )

    # ── Bug #2: Long files ────────────────────────────────────────────────────

    def test_long_file_produces_correct_chunk_count(self):
        """Bug #2: 100 lines at chunk_size=50 → exactly 2 chunks."""
        lines = [f"line {i}" for i in range(100)]
        result = _chunk_lines(lines, chunk_size=50)
        assert len(result) == 2

    def test_long_file_no_lines_lost(self):
        """Bug #2: Every line in input must appear in exactly one chunk."""
        lines = [f"line {i}" for i in range(73)]
        result = _chunk_lines(lines, chunk_size=20)
        all_lines = [line for chunk in result for line in chunk]
        assert all_lines == lines

    def test_remainder_lines_in_last_chunk(self):
        """Bug #2: Remaining lines after last full chunk appear in final chunk."""
        lines = [f"line {i}" for i in range(7)]
        result = _chunk_lines(lines, chunk_size=3)
        all_lines = [line for chunk in result for line in chunk]
        assert all_lines == lines


# ── _clean_model_output ───────────────────────────────────────────────────────

class TestCleanModelOutput:

    def test_removes_numeric_list_prefix_dot(self):
        raw = "1. Bạn hay nhầm lẫn giữa Interface và Abstract Class."
        result = _clean_model_output(raw)
        assert not result.startswith("1.")

    def test_removes_numeric_list_prefix_paren(self):
        raw = "2) Hãy cùng mình làm sáng tỏ mọi thứ."
        result = _clean_model_output(raw)
        assert not result.startswith("2)")

    def test_strips_blank_lines(self):
        raw = "Câu một.\n\n\nCâu hai.\n"
        result = _clean_model_output(raw)
        assert "\n\n" not in result

    def test_preserves_content(self):
        raw = "Interface là một contract trong Java, nó define method signature."
        result = _clean_model_output(raw)
        assert "Interface là một contract" in result

    def test_empty_input_returns_empty(self):
        assert _clean_model_output("") == ""

    def test_only_blank_lines_returns_empty(self):
        assert _clean_model_output("\n\n\n") == ""

    def test_mixed_numbered_and_normal(self):
        raw = "1. Câu đầu tiên.\nCâu không đánh số.\n2) Câu thứ ba."
        result = _clean_model_output(raw)
        lines = result.splitlines()
        assert all(not l.startswith(("1.", "2)", "1)", "2.")) for l in lines)


# ── process_file() integration (mocked Ollama) ────────────────────────────────

class TestProcessFile:
    """
    Integration test for process_file() with a fully mocked OllamaClient.
    No Ollama server required.
    """

    def _make_cfg(self, input_dir: Path, output_dir: Path):
        """Create a minimal ContextInferenceConfig without hitting .env."""
        from context_inference.config import ContextInferenceConfig

        cfg = ContextInferenceConfig.__new__(ContextInferenceConfig)
        cfg.input_dir = input_dir
        cfg.output_dir = output_dir
        cfg.min_words_per_line = 4
        cfg.chunk_size = 50
        cfg.request_timeout = 30
        cfg.temperature = 0.3
        cfg.model_name = "qwen2.5:3b"
        cfg.ollama_host = "http://localhost:11434"
        cfg.max_workers = 1
        return cfg

    def test_output_file_is_created(
        self,
        tmp_input_dir: Path,
        tmp_output_dir: Path,
    ):
        """process_file() must create an output file in output_dir."""
        cfg = self._make_cfg(tmp_input_dir, tmp_output_dir)
        mock_client = MagicMock()
        mock_client.infer_context.return_value = (
            "Bạn hay nhầm lẫn giữa Interface và Abstract Class trong Java?\n"
            "Hãy cùng mình làm sáng tỏ mọi thứ trong video này."
        )

        input_file = tmp_input_dir / "1-test.trans.txt"
        process_file(input_file, cfg, mock_client)

        expected = tmp_output_dir / "1-test-context.trans.txt"
        assert expected.exists(), f"Output file không được tạo: {expected}"

    def test_output_file_contains_model_response(
        self,
        tmp_input_dir: Path,
        tmp_output_dir: Path,
    ):
        """Content của output file phải chứa text từ model."""
        cfg = self._make_cfg(tmp_input_dir, tmp_output_dir)
        mock_client = MagicMock()
        expected_sentence = "Interface là một contract trong Java, nó define method signature."
        mock_client.infer_context.return_value = expected_sentence

        input_file = tmp_input_dir / "1-test.trans.txt"
        process_file(input_file, cfg, mock_client)

        output = (tmp_output_dir / "1-test-context.trans.txt").read_text(encoding="utf-8")
        assert expected_sentence in output

    def test_skips_unreadable_file(
        self,
        tmp_output_dir: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """process_file() không crash khi file input không đọc được — chỉ log lỗi."""
        cfg = self._make_cfg(tmp_path, tmp_output_dir)
        mock_client = MagicMock()

        non_existent = tmp_path / "ghost.trans.txt"
        with caplog.at_level(logging.ERROR):
            process_file(non_existent, cfg, mock_client)

        mock_client.infer_context.assert_not_called()
        assert "Không đọc được" in caplog.text or "ghost" in caplog.text

    def test_skips_all_short_lines_file(
        self,
        tmp_path: Path,
        tmp_output_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """File có toàn dòng quá ngắn → bỏ qua, không gọi model."""
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        short_file = input_dir / "short.trans.txt"
        short_file.write_text("OK\nJava\nTest\n", encoding="utf-8")

        cfg = self._make_cfg(input_dir, tmp_output_dir)
        mock_client = MagicMock()

        with caplog.at_level(logging.WARNING):
            process_file(short_file, cfg, mock_client)

        mock_client.infer_context.assert_not_called()

    def test_infer_called_once_per_chunk(
        self,
        tmp_path: Path,
        tmp_output_dir: Path,
    ):
        """Số lần gọi infer_context phải bằng số chunks."""
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        big_file = input_dir / "big.trans.txt"
        # 120 lines, each with ≥4 words → should produce 3 chunks of ~50
        content = "\n".join(
            [f"Đây là dòng số {i} trong file transcript dài" for i in range(120)]
        )
        big_file.write_text(content, encoding="utf-8")

        cfg = self._make_cfg(input_dir, tmp_output_dir)
        cfg.chunk_size = 50
        mock_client = MagicMock()
        mock_client.infer_context.return_value = "Câu được ghép lại."

        process_file(big_file, cfg, mock_client)

        # 120 lines / 50 = 3 chunks (no comma-ending lines so clean cuts)
        assert mock_client.infer_context.call_count == 3
