"""Tests for src/context_inference/processor.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from context_inference.processor import (  # noqa: E402
    _chunk_lines,
    _clean_model_output,
    _derive_output_path,
    process_file,
)


class TestDeriveOutputPath:
    def test_single_chunk_keeps_name(self, tmp_path: Path):
        p = tmp_path / "1-abc.trans.txt"
        out = _derive_output_path(p, tmp_path / "out", chunk_index=1, total_chunks=1)
        assert out.name == "1-abc-context.trans.txt"

    def test_multi_chunk_second_file_name(self, tmp_path: Path):
        p = tmp_path / "1-abc.trans.txt"
        out = _derive_output_path(p, tmp_path / "out", chunk_index=2, total_chunks=3)
        assert out.name == "1-2-abc-context.trans.txt"


class TestChunkLines:
    def test_empty_input(self):
        assert _chunk_lines([], 50) == []

    def test_150_lines_becomes_3_chunks(self):
        lines = [f"line {i}" for i in range(150)]
        chunks = _chunk_lines(lines, 50)
        assert len(chunks) == 3
        assert [len(c) for c in chunks] == [50, 50, 50]

    def test_170_lines_becomes_4_chunks(self):
        lines = [f"line {i}" for i in range(170)]
        chunks = _chunk_lines(lines, 50)
        assert len(chunks) == 4
        assert [len(c) for c in chunks] == [50, 50, 50, 20]

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError):
            _chunk_lines(["a"], 0)


class TestCleanModelOutput:
    def test_removes_meta_intro_line(self):
        raw = (
            "Duoi day la ket qua sau khi ghep cac doan text bi ngat:\n"
            "Khi ban tim kiem cong cu bao ve code Java."
        )
        cleaned = _clean_model_output(raw)
        assert "Duoi day" not in cleaned
        assert "Khi ban tim kiem" in cleaned

    def test_splits_sentences_to_one_line_each(self):
        raw = "Khi ban tim kiem cong cu bao ve code Java. Cac ban da san sang chua? Let's go!"
        cleaned = _clean_model_output(raw)
        lines = cleaned.splitlines()
        assert lines[0].endswith(".")
        assert lines[1].endswith("?")
        assert lines[2].endswith("!")


class TestProcessFile:
    def _make_cfg(self, input_dir: Path, output_dir: Path):
        from context_inference.config import ContextInferenceConfig

        cfg = ContextInferenceConfig.__new__(ContextInferenceConfig)
        cfg.input_dir = input_dir
        cfg.output_dir = output_dir
        cfg.min_words_per_line = 1
        cfg.chunk_size = 50
        cfg.request_timeout = 120
        cfg.temperature = 0.3
        cfg.model_name = "qwen2.5:3b"
        cfg.ollama_host = "http://localhost:11434"
        cfg.max_workers = 1
        return cfg

    def test_writes_one_file_per_chunk(self, tmp_path: Path):
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()

        src = input_dir / "1-test.trans.txt"
        src.write_text("\n".join([f"dong {i}" for i in range(150)]), encoding="utf-8")

        cfg = self._make_cfg(input_dir, output_dir)
        client = MagicMock()
        client.infer_context.return_value = "Noi dung da ghep."

        process_file(src, cfg, client)

        assert (output_dir / "1-test-context.trans.txt").exists()
        assert (output_dir / "1-2-test-context.trans.txt").exists()
        assert (output_dir / "1-3-test-context.trans.txt").exists()
        assert client.infer_context.call_count == 3
