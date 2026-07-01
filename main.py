"""Entry point for the YouTube transcript pipeline.

Usage:
    # Activate venv first
    .venv\\Scripts\\activate          # Windows
    source .venv/bin/activate        # Linux / macOS

    # Run the pipeline
    python main.py
"""

from finetune_sherpa_vi.config import load_config
from finetune_sherpa_vi.pipeline import Pipeline
from finetune_sherpa_vi.utils.logger import setup_logging


def main() -> None:
    # 1. Load config from .env
    config = load_config()

    # 2. Setup logging (must happen before any module uses get_logger)
    setup_logging(level=config.log_level)

    # 3. Run pipeline
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
