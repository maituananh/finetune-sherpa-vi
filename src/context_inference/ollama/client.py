"""Thin Ollama API client for context inference requests."""

from __future__ import annotations

import logging

import requests

from context_inference.config import ContextInferenceConfig
from context_inference.ollama.health import check_model_exists, check_ollama_running
from context_inference.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper around Ollama's ``/api/chat`` endpoint."""

    def __init__(self, cfg: ContextInferenceConfig) -> None:
        self.cfg = cfg
        self._validate_connection()

    def _validate_connection(self) -> None:
        if not check_ollama_running(self.cfg.ollama_host):
            raise RuntimeError(
                f"Ollama server không chạy tại {self.cfg.ollama_host}.\n"
                "Hãy chạy: ollama serve\n"
                "Hoặc kiểm tra biến OLLAMA_HOST trong .env"
            )

        if not check_model_exists(self.cfg.ollama_host, self.cfg.model_name):
            raise RuntimeError(
                f"Model '{self.cfg.model_name}' chưa được pull.\n"
                "Hãy chạy: bash scripts/download_model.sh"
            )

        logger.info("✅ Ollama server OK tại %s", self.cfg.ollama_host)
        logger.info("✅ Model '%s' sẵn sàng", self.cfg.model_name)

    def infer_context(self, lines: list[str]) -> str:
        """Send transcript fragments to the model and return cleaned plain text."""
        payload = {
            "model": self.cfg.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(lines)},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": 2048,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
            },
        }

        try:
            response = requests.post(
                f"{self.cfg.ollama_host}/api/chat",
                json=payload,
                timeout=self.cfg.request_timeout,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.error(
                "⏱️  Request timeout sau %ss — thử tăng CONTEXT_TIMEOUT trong .env",
                self.cfg.request_timeout,
            )
            raise
        except requests.exceptions.HTTPError as exc:
            logger.error("❌ Ollama API lỗi: %s", exc)
            raise
