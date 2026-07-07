"""
Configuration loader for the Context Inference module.
All settings are read from environment variables (loaded from .env).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ContextInferenceConfig:
    # ── Input / Output ──────────────────────────────────────────────────────
    # Folder containing .trans.txt files to process
    input_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CONTEXT_INPUT_DIR", "./transcripts")
        )
    )
    # Folder where *-context.trans.txt files will be written
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CONTEXT_OUTPUT_DIR", "./context_output")
        )
    )

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("CONTEXT_MODEL", "qwen2.5:3b")
    )

    # ── Processing ──────────────────────────────────────────────────────────
    # Number of files to process in parallel (keep ≤ 2 to avoid overload)
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_WORKERS", "2"))
    )
    # Minimum word count per line — lines below this threshold are skipped
    min_words_per_line: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_MIN_WORDS", "4"))
    )
    # Number of lines to feed the LLM per request (chunk size)
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_CHUNK_SIZE", "50"))
    )
    # Ollama request timeout in seconds
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_TIMEOUT", "120"))
    )

    # ── LLM generation params ───────────────────────────────────────────────
    temperature: float = field(
        default_factory=lambda: float(os.getenv("CONTEXT_TEMPERATURE", "0.3"))
    )

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"CONTEXT_INPUT_DIR không tồn tại: {self.input_dir.resolve()}"
            )
        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"CONTEXT_INPUT_DIR phải là folder: {self.input_dir.resolve()}"
            )


# Singleton instance
cfg = ContextInferenceConfig()
