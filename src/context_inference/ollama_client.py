"""Compatibility facade for the refactored Ollama integration."""

from context_inference.ollama.client import OllamaClient
from context_inference.ollama.health import (
    check_model_exists,
    check_ollama_running,
    log_gpu_info,
)

__all__ = [
    "OllamaClient",
    "check_model_exists",
    "check_ollama_running",
    "log_gpu_info",
]
