"""Batch orchestration for context inference files."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from context_inference.config import ContextInferenceConfig
from context_inference.ollama import OllamaClient, log_gpu_info
from context_inference.processing.file_processor import process_file

logger = logging.getLogger(__name__)


def run_batch(cfg: ContextInferenceConfig) -> None:
    """Process every transcript file in the configured input directory."""
    cfg.validate()

    logger.info("-- GPU Info ----------------------------------------------")
    log_gpu_info()
    logger.info("----------------------------------------------------------")

    txt_files = sorted(cfg.input_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("Khong tim thay file .txt nao trong: %s", cfg.input_dir.resolve())
        return

    logger.info(
        "Input : %s (%d file(s))\nOutput: %s\nModel : %s @ %s\nWorkers: %s",
        cfg.input_dir.resolve(),
        len(txt_files),
        cfg.output_dir.resolve(),
        cfg.model_name,
        cfg.ollama_host,
        cfg.max_workers,
    )

    client = OllamaClient(cfg)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {pool.submit(process_file, path, cfg, client): path for path in txt_files}

        total = len(futures)
        done = 0
        for future in as_completed(futures):
            source_file = futures[future]
            done += 1
            try:
                future.result()
            except Exception as exc:
                logger.error("[%s] Loi khong mong doi: %s", source_file.name, exc)
            logger.info("Tien do: %d/%d file(s) hoan thanh", done, total)

    logger.info("Hoan thanh toan bo batch!")
