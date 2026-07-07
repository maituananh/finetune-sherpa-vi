"""Real integration test (no mock) for context inference using a real transcript file."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from context_inference.config import ContextInferenceConfig  # noqa: E402
from context_inference.ollama_client import (  # noqa: E402
    OllamaClient,
    check_model_exists,
    check_ollama_running,
)
from context_inference.processor import process_file  # noqa: E402


@pytest.mark.integration
def test_real_inference_with_transcript_file(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[2]
    source_file = project_root / "transcripts" / "1-Hbp2zLIqmak.trans.txt"
    if not source_file.exists():
        pytest.skip(f"Missing input file: {source_file}")

    cfg = ContextInferenceConfig.__new__(ContextInferenceConfig)
    cfg.input_dir = source_file.parent
    cfg.output_dir = tmp_path / "context_output"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.min_words_per_line = 1
    cfg.chunk_size = 50
    cfg.request_timeout = 180
    cfg.temperature = 0.2
    cfg.model_name = "qwen2.5:3b"
    cfg.ollama_host = "http://localhost:11434"
    cfg.max_workers = 1

    if not check_ollama_running(cfg.ollama_host):
        pytest.skip("Ollama server is not running")
    if not check_model_exists(cfg.ollama_host, cfg.model_name):
        pytest.skip(f"Model not available: {cfg.model_name}")

    client = OllamaClient(cfg)
    process_file(source_file, cfg, client)

    out_files = sorted(cfg.output_dir.glob("1*-Hbp2zLIqmak-context.trans.txt"))
    assert len(out_files) >= 4, "170 input lines with chunk_size=50 should produce >=4 output files"

    raw_joined = "\n".join(f.read_text(encoding="utf-8") for f in out_files)
    lowered = raw_joined.lower()

    assert "duoi day la ket qua" not in lowered
    assert "dưới đây là kết quả" not in lowered

    lines = [ln.strip() for ln in raw_joined.splitlines() if ln.strip()]
    assert lines, "Output must contain at least one sentence"
    assert all(not re.match(r"^\d+[.)]\s*", ln) for ln in lines)
