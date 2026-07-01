"""
Tests for src/context_inference/config.py

Validates:
- Config loads correct defaults
- .env overrides are applied
- validate() raises on bad input paths
- output_dir is auto-created
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from context_inference.config import ContextInferenceConfig  # noqa: E402


class TestContextInferenceConfig:

    def test_default_model_name(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CONTEXT_MODEL", None)
            cfg = ContextInferenceConfig()
            assert cfg.model_name == "qwen2.5:3b"

    def test_env_override_model(self):
        with patch.dict(os.environ, {"CONTEXT_MODEL": "qwen2.5:7b"}):
            cfg = ContextInferenceConfig()
            assert cfg.model_name == "qwen2.5:7b"

    def test_default_max_workers(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CONTEXT_MAX_WORKERS", None)
            cfg = ContextInferenceConfig()
            assert cfg.max_workers == 2

    def test_env_override_max_workers(self):
        with patch.dict(os.environ, {"CONTEXT_MAX_WORKERS": "1"}):
            cfg = ContextInferenceConfig()
            assert cfg.max_workers == 1

    def test_output_dir_is_created(self, tmp_path: Path):
        out = tmp_path / "new_output"
        assert not out.exists()
        with patch.dict(os.environ, {"CONTEXT_OUTPUT_DIR": str(out)}):
            cfg = ContextInferenceConfig()
            assert cfg.output_dir.exists()

    def test_validate_raises_if_input_dir_missing(self, tmp_path: Path):
        with patch.dict(
            os.environ,
            {"CONTEXT_INPUT_DIR": str(tmp_path / "does_not_exist")},
        ):
            cfg = ContextInferenceConfig()
            with pytest.raises(FileNotFoundError):
                cfg.validate()

    def test_validate_raises_if_input_is_file_not_dir(self, tmp_path: Path):
        fake_file = tmp_path / "notadir.txt"
        fake_file.write_text("x")
        with patch.dict(os.environ, {"CONTEXT_INPUT_DIR": str(fake_file)}):
            cfg = ContextInferenceConfig()
            with pytest.raises(NotADirectoryError):
                cfg.validate()

    def test_validate_passes_with_valid_dir(self, tmp_path: Path):
        with patch.dict(os.environ, {"CONTEXT_INPUT_DIR": str(tmp_path)}):
            cfg = ContextInferenceConfig()
            cfg.validate()  # should not raise

    def test_default_ollama_host(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_HOST", None)
            cfg = ContextInferenceConfig()
            assert cfg.ollama_host == "http://localhost:11434"

    def test_temperature_is_float(self):
        cfg = ContextInferenceConfig()
        assert isinstance(cfg.temperature, float)

    def test_min_words_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CONTEXT_MIN_WORDS", None)
            cfg = ContextInferenceConfig()
            assert cfg.min_words_per_line == 4
