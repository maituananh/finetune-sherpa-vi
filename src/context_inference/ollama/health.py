"""Health checks and environment diagnostics for Ollama."""

from __future__ import annotations

import logging
import subprocess

import requests

logger = logging.getLogger(__name__)


def check_ollama_running(host: str) -> bool:
    """Return ``True`` when the Ollama server is reachable."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def check_model_exists(host: str, model_name: str) -> bool:
    """Return ``True`` when *model_name* is available in Ollama."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
        if response.status_code != 200:
            return False

        models = [model["name"] for model in response.json().get("models", [])]
        return any(model_name in model for model in models)
    except Exception:
        return False


def log_gpu_info() -> None:
    """Log best-effort GPU information for local debugging."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for index, gpu in enumerate(result.stdout.strip().split("\n")):
                name, total, free = [item.strip() for item in gpu.split(",")]
                logger.info("  GPU %d: %s | Total: %s | Free: %s", index, name, total, free)
            return

        logger.warning("  nvidia-smi không phản hồi — có thể đang chạy CPU only.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("  nvidia-smi không tìm thấy — chạy ở chế độ CPU.")
