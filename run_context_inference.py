#!/usr/bin/env python3
"""
Entry point: Context Inference Pipeline

Usage:
    python run_context_inference.py
    python run_context_inference.py --input ./my_transcripts --output ./my_output
    python run_context_inference.py --workers 1  # nếu máy yếu
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from context_inference import ContextInferenceConfig, run_batch  # noqa: E402


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Context Inference: ghép transcript fragments thành câu hoàn chỉnh."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Override CONTEXT_INPUT_DIR — folder chứa các file .txt cần xử lý.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Override CONTEXT_OUTPUT_DIR — folder lưu kết quả.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Override CONTEXT_MAX_WORKERS (khuyến nghị ≤ 2 để tránh quá tải).",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override CONTEXT_MODEL (mặc định: qwen2.5:3b).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (mặc định: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  Context Inference Pipeline — Qwen2.5:3B via Ollama")
    logger.info("=" * 60)

    # Build config (base from .env, overridden by CLI args)
    cfg = ContextInferenceConfig()
    if args.input:
        cfg.input_dir = args.input
    if args.output:
        cfg.output_dir = args.output
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers is not None:
        if args.workers > 3:
            logger.warning(
                f"⚠️  workers={args.workers} > 3 — có thể gây quá tải GPU/CPU. "
                "Khuyến nghị ≤ 2."
            )
        cfg.max_workers = args.workers
    if args.model:
        cfg.model_name = args.model

    try:
        run_batch(cfg)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("Hãy kiểm tra CONTEXT_INPUT_DIR trong file .env")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Bị dừng bởi người dùng (Ctrl+C).")
        sys.exit(0)


if __name__ == "__main__":
    main()
