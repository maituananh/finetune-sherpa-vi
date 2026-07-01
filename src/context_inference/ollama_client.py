"""
Ollama API client wrapper for the Context Inference module.
Handles connection checking, GPU detection info logging, and chat requests.
"""

import logging
import subprocess

import requests

from .config import ContextInferenceConfig
from .prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


def check_ollama_running(host: str) -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def log_gpu_info() -> None:
    """Log available GPU information for debugging across different machines."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split("\n")
            for i, gpu in enumerate(gpus):
                name, total, free = [x.strip() for x in gpu.split(",")]
                logger.info(f"  GPU {i}: {name} | Total: {total} | Free: {free}")
        else:
            logger.warning("  nvidia-smi không phản hồi — có thể đang chạy CPU only.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("  nvidia-smi không tìm thấy — chạy ở chế độ CPU.")


def check_model_exists(host: str, model_name: str) -> bool:
    """Return True if the model is already pulled in Ollama."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(model_name in m for m in models)
    except Exception:
        pass
    return False


class OllamaClient:
    """Thin wrapper around Ollama's /api/chat endpoint."""

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

        logger.info(f"✅ Ollama server OK tại {self.cfg.ollama_host}")
        logger.info(f"✅ Model '{self.cfg.model_name}' sẵn sàng")

    def infer_context(self, lines: list[str]) -> str:
        """
        Send a list of transcript lines to the LLM and return merged sentences.

        Args:
            lines: Raw fragment lines from the transcript file.

        Returns:
            Model output as a string (merged sentences, one per line).
        """
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
                "repeat_penalty": 1.1,
            },
        }

        try:
            resp = requests.post(
                f"{self.cfg.ollama_host}/api/chat",
                json=payload,
                timeout=self.cfg.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.error(
                f"⏱️  Request timeout sau {self.cfg.request_timeout}s — "
                "thử tăng CONTEXT_TIMEOUT trong .env"
            )
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Ollama API lỗi: {e}")
            raise
