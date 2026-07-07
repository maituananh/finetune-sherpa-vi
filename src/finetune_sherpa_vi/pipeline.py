"""Main pipeline orchestrator.

Wires together all modules and drives the end-to-end flow:

    For each playlist URL:
      1. Extract video IDs  (PlaylistExtractor)
      2. Filter already-processed IDs  (LockStore)
      3. Fetch transcript  (TranscriptFetcher)
      4. Write transcript to disk  (TranscriptWriter)
      5. Mark ID as processed  (LockStore)

Each step is logged so developers can follow progress in real time.
"""

from __future__ import annotations

from finetune_sherpa_vi.config import Config
from finetune_sherpa_vi.playlist.extractor import PlaylistExtractor
from finetune_sherpa_vi.storage.lock import LockStore
from finetune_sherpa_vi.storage.writer import TranscriptWriter
from finetune_sherpa_vi.transcript.fetcher import TranscriptFetcher
from finetune_sherpa_vi.utils.logger import get_logger

log = get_logger(__name__)


class Pipeline:
    """Runs the full YouTube → transcript pipeline."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._extractor = PlaylistExtractor(video_limit=config.playlist_video_limit)
        self._fetcher = TranscriptFetcher(languages=config.transcript_langs)
        self._lock = LockStore(config.lock_file)
        self._writer = TranscriptWriter(config.output_dir)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Execute the pipeline for all configured playlists."""
        cfg = self._config

        if not cfg.playlists:
            log.warning("[Pipeline] No playlists configured. Set YOUTUBE_PLAYLISTS in .env")
            return

        log.info("=" * 60)
        log.info("[Pipeline] Starting — %d playlist(s) to process", len(cfg.playlists))
        log.info("[Pipeline] Output dir : %s", cfg.output_dir.resolve())
        log.info("[Pipeline] Lock file  : %s", cfg.lock_file.resolve())
        log.info("[Pipeline] Languages  : %s", cfg.transcript_langs)
        log.info("[Pipeline] Playlist limit : %s", cfg.playlist_video_limit or "all")
        log.info("=" * 60)

        total_processed = 0
        total_skipped = 0
        total_failed = 0

        for playlist_idx, playlist_url in enumerate(cfg.playlists, start=1):
            log.info("")
            log.info(
                "── Playlist %d/%d ──────────────────────────────────────────",
                playlist_idx,
                len(cfg.playlists),
            )
            processed, skipped, failed = self._process_playlist(playlist_url)
            total_processed += processed
            total_skipped += skipped
            total_failed += failed

        log.info("")
        log.info("=" * 60)
        log.info("[Pipeline] Done!")
        log.info("  Processed : %d", total_processed)
        log.info("  Skipped   : %d (already in lock file)", total_skipped)
        log.info("  Failed    : %d (no transcript / error)", total_failed)
        log.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _process_playlist(self, playlist_url: str) -> tuple[int, int, int]:
        """Process a single playlist; return (processed, skipped, failed) counts."""
        try:
            videos = self._extractor.extract(playlist_url)
        except Exception as exc:  # noqa: BLE001
            log.error("[Playlist] ERROR extracting playlist — %s: %s", playlist_url, exc)
            return 0, 0, 0

        processed = skipped = failed = 0

        for video in videos:
            vid_id = video.video_id
            title = video.title

            # ── Step 1: Check lock ─────────────────────────────────────
            if self._lock.is_locked(vid_id):
                log.warning(
                    "[Lock] SKIP '%s' (%s) — already processed (in lock file)",
                    title,
                    vid_id,
                )
                skipped += 1
                continue

            log.info("[Video] Processing: '%s' (%s)", title, vid_id)

            # ── Step 2: Fetch transcript ───────────────────────────────
            transcript_result = self._fetcher.fetch(vid_id)
            segments = transcript_result.segments

            if segments is None:
                failure_reason = transcript_result.reason or "No transcript obtained."
                log.warning(
                    "[Video] FAIL '%s' (%s) — %s",
                    title,
                    vid_id,
                    failure_reason,
                )
                failed += 1
                # Still lock it so we don't waste time retrying on next run.
                # Remove this line if you prefer to retry failed videos.
                self._lock.lock(vid_id)
                continue

            # ── Step 3: Write to disk ──────────────────────────────────
            self._writer.write(video_id=vid_id, segments=segments)

            # ── Step 4: Mark as done ───────────────────────────────────
            self._lock.lock(vid_id)
            processed += 1

        return processed, skipped, failed
