# finetune-sherpa-vi

Fine-tuning Sherpa Vietnamese speech model.

## Project Structure

```
finetune-sherpa-vi/
├── main.py                               # YouTube transcript entry point
├── run_context_inference.py              # Context inference pipeline entry point
├── scripts/
│   └── download_model.sh                 # Install Ollama & pull Qwen2.5:3B
├── src/
│   ├── finetune_sherpa_vi/
│   │   ├── config.py                     # Load .env → typed Config (YouTube)
│   │   ├── pipeline.py                   # Main orchestrator (YouTube)
│   │   ├── playlist/
│   │   │   └── extractor.py              # Extract video IDs from playlists
│   │   ├── transcript/
│   │   │   └── fetcher.py                # Fetch captions
│   │   ├── storage/
│   │   │   ├── lock.py                   # Track processed IDs
│   │   │   └── writer.py                 # Write {n}-{ID}.trans.txt files
│   │   └── utils/
│   │       └── logger.py                 # Centralized logger factory
│   └── context_inference/
│       ├── __init__.py
│       ├── config.py                     # Load .env → typed Config (LLM context)
│       ├── prompt.py                     # Expert IT Prompt (Vi+En code-switching)
│       ├── ollama/
│       │   ├── client.py                 # Ollama chat client
│       │   └── health.py                 # Ollama/server/model/GPU checks
│       ├── processing/
│       │   ├── batch.py                  # Batch orchestration
│       │   ├── file_processor.py         # Single-file processing flow
│       │   ├── naming.py                 # Output filename rules
│       │   └── text.py                   # Filter/chunk/cleanup helpers
│       ├── ollama_client.py              # Compatibility wrapper
│       └── processor.py                  # Compatibility wrapper
├── transcripts/                          # (auto-created) output .trans.txt files
├── context_output/                       # (auto-created) output of processed context transcripts
├── video-id-locked.txt                   # (auto-created) processed video IDs
├── tests/
│   ├── __init__.py
│   └── context_inference/                # Test suite for Context Inference
│       ├── __init__.py
│       ├── conftest.py                   # Shared pytest fixtures
│       ├── test_config.py                # Tests for configuration and env validation
│       ├── test_processor.py             # Tests for smart chunking (comma guard) & processing
│       ├── test_prompt.py                # Tests for prompt integrity & structure
│       └── transcripts/
│           └── 1-test.trans.txt          # IT transcript sample for testing
├── .env                                  # (gitignored) local env vars
├── .env.example                          # Template — commit this
├── .flake8                               # Flake8 config
└── pyproject.toml                        # Project metadata + tool config
```

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate

# 3. Activate (Linux/macOS)
source .venv/bin/activate

# 4. Install all dependencies (including dev and context-inference deps)
pip install -e ".[dev,context-inference]"

# 5. Copy env template and configure
cp .env.example .env
# → Edit .env and adjust variables
```

---

## Step-by-Step Running Guide

To run the entire pipeline from scratch, follow these steps in order:

### 1. Environment & Config Setup
Ensure you have activated your virtual environment and installed the dependencies:
```bash
source .venv/bin/activate
pip install -e ".[dev,context-inference]"
cp .env.example .env
```
Open `.env` and configure:
- `YOUTUBE_PLAYLISTS`: The playlist URLs containing videos to download.
- `YOUTUBE_PLAYLIST_VIDEO_LIMIT`: Number of videos to take from each playlist. Leave empty or set `0` to take all.
- `CONTEXT_INPUT_DIR` / `TRANSCRIPT_OUTPUT_DIR`: Keep them matching (e.g., `./transcripts`) so the LLM knows where to find the raw text.

### 2. Extract Youtube Transcripts
Extract the raw timestamp segment files from YouTube:
```bash
python main.py
```
This fetches raw segments and saves them to the `./transcripts` directory.

### 3. Setup Ollama Server & Pull Model
Initialize the local LLM environment (installs Ollama, starts background server, pulls Qwen2.5:3B, and runs sanity checks):
```bash
bash scripts/download_model.sh
```
Ensure you see a success output: `✅ Mọi thứ đã sẵn sàng!`

### 4. Run Context Inference
Process the raw segment files to reconstruct coherent sentences:
```bash
python run_context_inference.py
```
This will read from `./transcripts`, run context inference, filter short lines, and write files to `./context_output/` with name format: `*-context.trans.txt`.

---


## 1. YouTube Transcript Pipeline

### Configuration (.env)

| Variable | Description | Example |
|---|---|---|
| `YOUTUBE_PLAYLISTS` | JSON array of playlist URLs | `["https://www.youtube.com/playlist?list=PL..."]` |
| `YOUTUBE_PLAYLIST_VIDEO_LIMIT` | Max videos to take from each playlist; `0` or empty = all | `10` |
| `TRANSCRIPT_LANGS` | Language priority (comma-separated) | `vi,en` |
| `TRANSCRIPT_OUTPUT_DIR` | Where to save .trans.txt files | `./transcripts` |
| `LOCK_FILE` | Processed video IDs lock file | `./video-id-locked.txt` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

### Running the Pipeline

```bash
python main.py
```

---

## 2. Context Inference Pipeline (LLM)

This module reads raw segment transcripts, skips very short lines, and uses **Qwen2.5:3B via Ollama** to rebuild fragmented lines into complete, grammatically correct sentences matching the speech length of ~15s (about 30-35 words per line). 

It is tailored for Vietnamese IT content, preserving technical English terms (code-switching) and avoiding splitting context mid-sentence (e.g. avoiding splits after comma-ending lines).

The code is organized by responsibility so it is easier to replace pieces independently:
- `context_inference/ollama/`: network and runtime checks for Ollama.
- `context_inference/processing/`: pure text transforms, file processing, and batch orchestration.
- `context_inference/prompt.py`: prompt rules only.
- `context_inference/config.py`: env-backed settings only.

### Setup Ollama & Pull Model

Run the download script which will install Ollama (if not present), launch the server, pull the `qwen2.5:3b` model, and inspect active GPU/VRAM:

```bash
bash scripts/download_model.sh
```

### Configuration (.env)

Add the following context variables to your `.env` file:

| Variable | Description | Default |
|---|---|---|
| `CONTEXT_INPUT_DIR` | Directory containing input `.txt` files | `./transcripts` |
| `CONTEXT_OUTPUT_DIR` | Directory to save processed context files | `./context_output` |
| `CONTEXT_MODEL` | Ollama model name | `qwen2.5:3b` |
| `CONTEXT_MAX_WORKERS` | Max parallel files to process (keep <= 2) | `2` |
| `CONTEXT_MIN_WORDS` | Skip lines containing fewer than this word count | `4` |
| `CONTEXT_CHUNK_SIZE` | Chunk size (lines) sent to LLM per request | `50` |
| `CONTEXT_TIMEOUT` | Ollama connection timeout (seconds) | `120` |
| `CONTEXT_TEMPERATURE` | Creativity degree of LLM output | `0.3` |
| `OLLAMA_HOST` | Ollama API endpoint host | `http://localhost:11434` |

### Running the Pipeline

```bash
python run_context_inference.py
```

You can also override configs directly via CLI:
```bash
python run_context_inference.py --input ./my_transcripts --output ./my_output --workers 1
```

*Output files are written as: `<original_name>-context.trans.txt` in the output folder.*

---

## Testing & Quality Control

### Running Local Tests

We use `pytest` for unit and integration testing. All context inference tests mock out the Ollama client to run offline instantly:

```bash
python3 -m pytest tests/context_inference/ -v
```

### Code Formatting and Linting

```bash
black .                  # Auto-format code styles
isort .                  # Auto-sort import blocks
flake8 .                 # Check code styles and warnings (defined in .flake8)
pyright                  # Static type analyzer
```

### Continuous Integration (GitHub Actions)

When you push code or open a pull request, the CI pipeline (`.github/workflows/ci.yml`) automatically executes:
1. **Format Check**: runs `black` and `isort` dry-runs.
2. **Linter**: runs `flake8`.
3. **Type Checker**: runs `pyright`.
4. **Test Suite**: runs `pytest` and generates a coverage summary.
5. **Env Validation**: ensures `.env.example` has all required settings.
6. **Script Validation**: checks shell scripts via `shellcheck`.
7. **Security scan**: run `pip-audit` for vulnerability scans.
