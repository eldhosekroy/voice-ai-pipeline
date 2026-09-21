"""
Entry point for the modular Voice AI Pipeline application.
"""

import logging
import sys
from config.config import get_config
from src.pipeline.pipeline import VoicePipeline

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("voice-ai-pipeline")


def main() -> None:
    """Initialize configuration and run the Voice AI Pipeline."""
    logger.info("Initializing Voice AI Pipeline project (STEP 1)...")

    # Load configuration settings from environment / defaults
    config = get_config()
    logger.info("Configuration loaded:")
    logger.info("  OLLAMA_MODEL: %s", config.ollama_model)
    logger.info("  WHISPER_MODEL: %s", config.whisper_model)
    logger.info("  WHISPER_DEVICE: %s", config.whisper_device)
    logger.info("  KOKORO_VOICE: %s", config.kokoro_voice)

    # Initialize modular voice pipeline
    pipeline = VoicePipeline(config=config)

    # TODO: Execute pipeline run loop when audio input & model components are ready
    pipeline.run()

    logger.info("STEP 1 skeleton initialization complete.")


if __name__ == "__main__":
    main()
