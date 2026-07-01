# finetune-sherpa-vi

Fine-tuning Sherpa Vietnamese speech model.

## Project Structure

```
finetune-sherpa-vi/
├── main.py                               # Entry point
├── src/
│   └── finetune_sherpa_vi/
│       ├── config.py                     # Load .env → typed Config
│       ├── pipeline.py                   # Main orchestrator
│       ├── playlist/
│       │   └── extractor.py              # Extract video IDs from playlists (yt-dlp)
│       ├── transcript/
│       │   └── fetcher.py                # Fetch captions (youtube-transcript-api)
│       ├── storage/
│       │   ├── lock.py                   # Track processed IDs (video-id-locked.txt)
│       │   └── writer.py                 # Write {n}-{ID}.trans.txt files
│       └── utils/
│           └── logger.py                 # Centralized logger factory
├── transcripts/                          # (auto-created) output .trans.txt files
├── video-id-locked.txt                   # (auto-created) processed video IDs
├── .env                                  # (gitignored) local env vars
├── .env.example                          # Template — commit this
├── .flake8                               # Flake8 config
├── pyproject.toml                        # Project metadata + tool config
└── README.md
```

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate

# 3. Activate (Linux/macOS)
source .venv/bin/activate

# 4. Install all dependencies
pip install -e ".[dev]"

# 5. Copy env template and configure
cp .env.example .env
# → Edit .env and set YOUTUBE_PLAYLISTS
```

## Configuration (.env)

| Variable | Description | Example |
|---|---|---|
| `YOUTUBE_PLAYLISTS` | JSON array of playlist URLs | `["https://www.youtube.com/playlist?list=PL..."]` |
| `TRANSCRIPT_LANGS` | Language priority (comma-separated) | `vi,en` |
| `TRANSCRIPT_OUTPUT_DIR` | Where to save .trans.txt files | `./transcripts` |
| `LOCK_FILE` | Processed video IDs lock file | `./video-id-locked.txt` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## Running the Pipeline

```bash
# Activate venv first, then:
python main.py
```

Sample output:
```
10:05:01 [INFO] pipeline: ============================================================
10:05:01 [INFO] pipeline: [Pipeline] Starting — 2 playlist(s) to process
10:05:01 [INFO] pipeline: [Pipeline] Output dir : D:\...\transcripts
10:05:01 [INFO] pipeline: ── Playlist 1/2 ──────────────────────────────────
10:05:02 [INFO] playlist.extractor: [Playlist] Found 42 videos
10:05:02 [INFO] pipeline: [Video] Processing: 'Học tiếng Việt - Bài 1' (dQw4w9WgXcQ)
10:05:03 [INFO] transcript.fetcher: [Transcript] OK — 128 segments retrieved
10:05:03 [INFO] storage.writer:   → Saved transcript: 1-dQw4w9WgXcQ.trans.txt (128 segments)
10:05:03 [WARNING] pipeline: [Lock] SKIP 'Bài 2' (9bZkp7q19f0) — already processed
...
10:06:00 [INFO] pipeline: [Pipeline] Done!
10:06:00 [INFO] pipeline:   Processed : 35
10:06:00 [INFO] pipeline:   Skipped   : 6 (already in lock file)
10:06:00 [INFO] pipeline:   Failed    : 1 (no transcript / error)
```

## Output Files

- **`transcripts/{n}-{video_id}.trans.txt`** — One file per video; each line is a transcript segment.
- **`video-id-locked.txt`** — One video ID per line; videos here are skipped on subsequent runs.

## Code Quality

```bash
black .      # Format
isort .      # Sort imports
flake8 .     # Lint
pyright      # Type check
```
