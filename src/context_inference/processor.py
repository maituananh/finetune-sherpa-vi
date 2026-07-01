"""
Core file processor for the Context Inference module.

Processing pipeline per file:
  1. Read all lines from the .trans.txt file
  2. Filter / skip lines that are too short (log warning for devs)
  3. Split into smart chunks — never break at a comma-ending line (bug fix #1)
  4. Send each chunk to Ollama → get back merged sentences
  5. Write output to <output_dir>/<stem>-context.trans.txt

Parallelism: ThreadPoolExecutor with max_workers=CONTEXT_MAX_WORKERS (default 2)
to keep hardware load manageable.

Bug fixes applied:
  #1 Lines ending with ',' are never placed at the END of a chunk — they are
     carried over to the next chunk so the model sees the continuation.
  #2 chunk_size is treated as a soft ceiling; we always finish the current
     logical group before cutting, preventing sentences from being split in half.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import ContextInferenceConfig
from .ollama_client import OllamaClient, log_gpu_info

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _derive_output_path(input_path: Path, output_dir: Path) -> Path:
    """
    Build the output file path.

    Example:
        input:  /transcripts/1-abc-xyz.trans.txt
        output: /context_output/1-abc-xyz-context.trans.txt
    """
    stem = input_path.name  # e.g. "1-abc-xyz.trans.txt"

    # Insert "-context" before the first dot
    if "." in stem:
        base, *exts = stem.split(".")
        new_name = base + "-context." + ".".join(exts)
    else:
        new_name = stem + "-context"

    return output_dir / new_name


def _filter_lines(
    raw_lines: list[str], min_words: int, source_file: str
) -> list[str]:
    """
    Return only lines that meet the minimum word count.
    Short lines are logged so developers can review them.
    """
    kept: list[str] = []
    skipped: list[str] = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue  # silently drop blank lines
        word_count = len(stripped.split())
        if word_count < min_words:
            skipped.append(stripped)
        else:
            kept.append(stripped)

    if skipped:
        logger.warning(
            f"[{source_file}] ⚠️  Bỏ qua {len(skipped)} dòng quá ngắn (< {min_words} từ):"
        )
        for s in skipped:
            logger.warning(f"  SKIPPED → \"{s}\"")

    return kept


def _chunk_lines(lines: list[str], chunk_size: int) -> list[list[str]]:
    """Split lines into chunks, never cutting at a comma-ending line (bug #1 & #2).

    A line that ends with ',' is part of an unfinished sentence — it must stay
    together with its continuation in the next chunk.  We therefore delay the
    chunk boundary until we hit a line that does NOT end with ','.

    Args:
        lines: Pre-filtered list of transcript lines.
        chunk_size: Soft maximum number of lines per chunk.

    Returns:
        List of chunks, each a list of lines.
    """
    if not lines:
        return []

    chunks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        current.append(line)
        # Only close the chunk when:
        #   a) we have reached chunk_size, AND
        #   b) the last line does NOT end with ',' (unfinished sentence guard)
        if len(current) >= chunk_size and not line.rstrip().endswith(","):
            chunks.append(current)
            current = []

    if current:  # flush the last (possibly smaller) chunk
        chunks.append(current)

    return chunks


def _clean_model_output(raw: str) -> str:
    """
    Lightly clean model output:
    - Remove any leading numbering (1. 2. etc.) added despite instructions
    - Strip extra blank lines
    """
    cleaned_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove leading numbering like "1. " or "1) "
        line = re.sub(r"^\d+[.)]\s*", "", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# ── Per-file processor ────────────────────────────────────────────────────────

def process_file(input_path: Path, cfg: ContextInferenceConfig, client: OllamaClient) -> None:
    """
    Process a single .trans.txt file: filter → chunk → infer → write output.
    This function is designed to be called from a thread pool.
    """
    file_label = input_path.name
    logger.info(f"🔄 Bắt đầu xử lý: {file_label}")

    # 1. Read
    try:
        raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        logger.error(f"❌ [{file_label}] Không đọc được file: {e}")
        return

    # 2. Filter short lines
    valid_lines = _filter_lines(raw_lines, cfg.min_words_per_line, file_label)

    if not valid_lines:
        logger.warning(f"⚠️  [{file_label}] Không còn dòng nào sau khi lọc — bỏ qua file.")
        return

    # 3. Chunk
    chunks = _chunk_lines(valid_lines, cfg.chunk_size)
    logger.info(
        f"  [{file_label}] {len(valid_lines)} dòng hợp lệ → {len(chunks)} chunk(s) gửi model"
    )

    # 4. Infer each chunk sequentially (one GPU — no benefit in parallel API calls)
    output_parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        logger.info(f"  [{file_label}] Chunk {idx}/{len(chunks)} ({len(chunk)} dòng)…")
        try:
            result = client.infer_context(chunk)
            cleaned = _clean_model_output(result)
            output_parts.append(cleaned)
        except Exception as e:
            logger.error(f"  [{file_label}] ❌ Chunk {idx} thất bại: {e}")
            # Keep going with remaining chunks instead of aborting entire file
            output_parts.append(f"# [LỖI CHUNK {idx}] {e}")

    # 5. Write
    output_path = _derive_output_path(input_path, cfg.output_dir)
    try:
        output_path.write_text("\n\n".join(output_parts) + "\n", encoding="utf-8")
        logger.info(f"✅ [{file_label}] → {output_path.name}")
    except Exception as e:
        logger.error(f"❌ [{file_label}] Không ghi được output: {e}")


# ── Batch processor (entry point) ─────────────────────────────────────────────

def run_batch(cfg: ContextInferenceConfig) -> None:
    """
    Discover all .txt files in cfg.input_dir and process them in a
    bounded thread pool (max_workers from config).
    """
    cfg.validate()

    # Log GPU info so dev can verify which GPU is active
    logger.info("── GPU Info ──────────────────────────────────────────────")
    log_gpu_info()
    logger.info("──────────────────────────────────────────────────────────")

    txt_files = sorted(cfg.input_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"⚠️  Không tìm thấy file .txt nào trong: {cfg.input_dir.resolve()}")
        return

    logger.info(
        f"📂 Input : {cfg.input_dir.resolve()} ({len(txt_files)} file(s))\n"
        f"📁 Output: {cfg.output_dir.resolve()}\n"
        f"🤖 Model : {cfg.model_name} @ {cfg.ollama_host}\n"
        f"⚙️  Workers: {cfg.max_workers}"
    )

    # Initialise client once — validates connection before spawning threads
    client = OllamaClient(cfg)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(process_file, f, cfg, client): f
            for f in txt_files
        }

        total = len(futures)
        done = 0
        for future in as_completed(futures):
            src_file = futures[future]
            done += 1
            try:
                future.result()  # re-raise any unhandled exception
            except Exception as e:
                logger.error(f"❌ [{src_file.name}] Lỗi không mong đợi: {e}")
            logger.info(f"📊 Tiến độ: {done}/{total} file(s) hoàn thành")

    logger.info("🎉 Hoàn thành toàn bộ batch!")
