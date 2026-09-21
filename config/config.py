"""
Configuration module for the voice AI pipeline.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration settings for the Voice AI Pipeline components."""

    # VAD (Silero) Settings
    vad_device: str = os.getenv("VAD_DEVICE", "cpu")
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    vad_min_speech_duration: float = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.25"))
    vad_min_silence_duration: float = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.5"))

    # STT (Whisper) Settings
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # LLM (Ollama / Qwen) Settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5")

    # TTS (Kokoro) Settings
    kokoro_voice: str = os.getenv("KOKORO_VOICE", "af_heart")
    kokoro_device: str = os.getenv("KOKORO_DEVICE", "cpu")

    # Audio Settings
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))


def get_config() -> PipelineConfig:
    """Helper function to return a fresh PipelineConfig instance."""
    return PipelineConfig()
