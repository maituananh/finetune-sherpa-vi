# finetune-sherpa-vi

Fine-tuning Sherpa Vietnamese speech model.

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate

# 3. Activate (Linux/macOS)
source .venv/bin/activate

# 4. Install dev dependencies
pip install -e ".[dev]"

# 5. Copy env template
cp .env.example .env
# → Edit .env and fill in your API keys
```

## Code Quality

```bash
# Format
black .

# Sort imports
isort .

# Lint
flake8 .

# Type check
pyright
```

## Project Structure

```
finetune-sherpa-vi/
├── src/
│   └── finetune_sherpa_vi/   # Main package
│       └── __init__.py
├── tests/                    # Unit tests
├── data/                     # (gitignored) raw & processed data
├── models/                   # (gitignored) model checkpoints
├── output/                   # (gitignored) training outputs
├── .env                      # (gitignored) local env vars
├── .env.example              # Env var template — commit this
├── .flake8                   # Flake8 config
├── .gitignore
├── pyproject.toml            # Project metadata + tool config
└── README.md
```
