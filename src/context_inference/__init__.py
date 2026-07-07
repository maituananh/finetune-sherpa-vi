"""Context inference pipeline package."""

from context_inference.config import ContextInferenceConfig
from context_inference.ollama import OllamaClient, check_model_exists, check_ollama_running
from context_inference.processing import process_file, run_batch

__all__ = [
    "ContextInferenceConfig",
    "OllamaClient",
    "check_model_exists",
    "check_ollama_running",
    "process_file",
    "run_batch",
]
