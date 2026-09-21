"""
Speech-to-Text (STT) component interface and Whisper implementation.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseSTT(ABC):
    """Abstract interface for Speech-to-Text engines."""

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio bytes into text.

        :param audio_data: Raw audio data
        :return: Transcribed text string
        """
        pass


class WhisperSTT(BaseSTT):
    """Whisper Speech-to-Text implementation stub."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        # TODO: Initialize Whisper model (e.g., faster-whisper or openai-whisper)
        logger.info(
            "[STT] Initialized WhisperSTT stub (model=%s, device=%s, compute_type=%s)",
            self.model_name,
            self.device,
            self.compute_type,
        )

    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio into text using Whisper model.
        """
        # TODO: Implement Whisper STT model transcription
        # 1. Convert audio bytes to supported audio array/file format
        # 2. Run model.transcribe()
        # 3. Return output text
        logger.info("[STT] Transcribing audio...")
        return ""
