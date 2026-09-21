"""
Audio output module for playing audio through system speakers using sounddevice.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger(__name__)


class BaseAudioOutput(ABC):
    """Abstract interface for audio output devices."""

    @abstractmethod
    def play(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """
        Play audio data through speakers.

        :param audio_data: Audio samples (1D numpy float32 or int16 array)
        :param sample_rate: Audio sampling rate in Hz
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop playback immediately."""
        pass


class Speaker(BaseAudioOutput):
    """
    Speaker audio output implementation using sounddevice.
    """

    def __init__(self, device: int | str | None = None) -> None:
        """
        Initialize Speaker output device.

        :param device: Optional output device index or name.
        """
        self.device = device
        self._is_playing: bool = False

    def play(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """
        Play floating point audio data through system speaker.
        """
        if sd is None:
            logger.error("[AUDIO] sounddevice is not available. Playback skipped.")
            return

        if audio_data is None or len(audio_data) == 0:
            logger.warning("[AUDIO] Received empty audio buffer for playback.")
            return

        try:
            logger.info("[AUDIO] Playing response (%d samples, %d Hz)...", len(audio_data), sample_rate)
            self._is_playing = True
            sd.play(audio_data, samplerate=sample_rate, device=self.device)
            sd.wait()  # Wait until playback completes
            logger.info("[AUDIO] Audio playback finished.")
        except Exception as e:
            logger.error("[AUDIO] Speaker playback error: %s", e)
        finally:
            self._is_playing = False

    def stop(self) -> None:
        """Stop current playback."""
        if sd is not None and self._is_playing:
            try:
                sd.stop()
                logger.info("[AUDIO] Speaker playback stopped.")
            except Exception as e:
                logger.error("[AUDIO] Error stopping speaker playback: %s", e)
            finally:
                self._is_playing = False
