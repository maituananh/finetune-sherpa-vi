"""Compatibility facade for the refactored context inference pipeline."""

from context_inference.processing.batch import run_batch
from context_inference.processing.file_processor import process_file
from context_inference.processing.naming import derive_output_path as _derive_output_path
from context_inference.processing.text import (
    chunk_lines as _chunk_lines,
    clean_model_output as _clean_model_output,
)

__all__ = [
    "run_batch",
    "process_file",
    "_derive_output_path",
    "_chunk_lines",
    "_clean_model_output",
]
