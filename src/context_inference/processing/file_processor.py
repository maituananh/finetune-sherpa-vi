"""Single-file processing flow for context inference."""

from __future__ import annotations

import logging
from pathlib import Path

from context_inference.config import ContextInferenceConfig
from context_inference.ollama import OllamaClient
from context_inference.processing.naming import derive_output_path
from context_inference.processing.text import chunk_lines, clean_model_output, filter_lines

logger = logging.getLogger(__name__)


def process_file(input_path: Path, cfg: ContextInferenceConfig, client: OllamaClient) -> None:
    """Process one transcript file and write one output file per chunk."""
    file_label = input_path.name
    logger.info("Bat dau xu ly: %s", file_label)

    try:
        raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.error("[%s] Khong doc duoc file: %s", file_label, exc)
        return

    valid_lines = filter_lines(raw_lines, cfg.min_words_per_line, file_label)
    if not valid_lines:
        logger.warning("[%s] Khong con dong nao sau khi loc - bo qua file.", file_label)
        return

    chunks = chunk_lines(valid_lines, cfg.chunk_size)
    logger.info(
        "[%s] %d dong hop le -> %d chunk(s) gui model",
        file_label,
        len(valid_lines),
        len(chunks),
    )

    for index, chunk in enumerate(chunks, start=1):
        logger.info("[%s] Chunk %d/%d (%d dong)...", file_label, index, len(chunks), len(chunk))
        output_path = derive_output_path(
            input_path,
            cfg.output_dir,
            chunk_index=index,
            total_chunks=len(chunks),
        )

        try:
            result = client.infer_context(chunk)
            cleaned = clean_model_output(result)
            output_path.write_text(cleaned + "\n", encoding="utf-8")
            logger.info("[%s] Chunk %d -> %s", file_label, index, output_path.name)
        except Exception as exc:
            logger.error("[%s] Chunk %d that bai: %s", file_label, index, exc)
            try:
                output_path.write_text(f"# [LOI CHUNK {index}] {exc}\n", encoding="utf-8")
            except Exception as write_error:
                logger.error("[%s] Khong ghi duoc output chunk %d: %s", file_label, index, write_error)
