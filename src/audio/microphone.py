"""
Microphone audio capture module using sounddevice.
Provides a clean abstraction for capturing real-time mono audio streams.
"""

from abc import ABC, abstractmethod
import logging
import queue
from typing import Any
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # Handled gracefully if dependency not installed yet

logger = logging.getLogger(__name__)


class BaseAudioInput(ABC):
    """Abstract interface for audio input devices."""

    @abstractmethod
    def start(self) -> None:
        """Start capturing audio."""
        pass

    @abstractmethod
    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Read a chunk of audio data.

        :param timeout: Maximum time in seconds to wait for data.
        :return: 1D numpy array of audio samples (float32) or None if timeout.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release all audio stream resources."""
        pass


class Microphone(BaseAudioInput):
    """
    Microphone capture implementation using sounddevice.
    Captures mono audio into a thread-safe queue.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        channels: int = 1,
        dtype: str = "float32",
        device: int | str | None = None,
    ) -> None:
        """
        Initialize Microphone instance.

        :param sample_rate: Audio sampling rate in Hz (default: 16000).
        :param chunk_size: Number of frames per block/chunk (default: 512).
        :param channels: Number of audio channels (1 for mono).
        :param dtype: Audio sample data type ('float32' or 'int16').
        :param device: Optional input device index or name.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        self.device = device

        self._stream: Any = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_running: bool = False

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Callback invoked by sounddevice for each audio block."""
        if status:
            logger.warning("[AUDIO] Stream status warning: %s", status)

        if self._is_running:
            # Flatten mono input array to 1D float32 array
            chunk = indata.copy().flatten()
            self._audio_queue.put(chunk)

    def start(self) -> None:
        """Start recording from microphone."""
        if sd is None:
            raise RuntimeError(
                "sounddevice package is not installed. Please run `pip install sounddevice numpy`."
            )

        if self._is_running:
            logger.debug("[AUDIO] Microphone stream is already running.")
            return

        try:
            logger.info(
                "[AUDIO] Starting microphone stream (sample_rate=%d, chunk_size=%d, channels=%d)...",
                self.sample_rate,
                self.chunk_size,
                self.channels,
            )
            # Empty queue from previous runs
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
            logger.info("[AUDIO] Microphone stream started successfully.")
        except Exception as e:
            logger.error("[AUDIO] Failed to start microphone stream: %s", e)
            self._is_running = False
            raise

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Read the next audio chunk from queue.

        :param timeout: Time to wait in seconds before returning None.
        :return: 1D numpy array of audio samples or None.
        """
        if not self._is_running:
            logger.warning("[AUDIO] Cannot read chunk: Microphone stream is not running.")
            return None

        try:
            return self._audio_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            logger.debug("[AUDIO] Read chunk timed out after %.2f seconds.", timeout)
            return None

    def stop(self) -> None:
        """Stop recording audio stream."""
        if not self._is_running:
            return

        logger.info("[AUDIO] Stopping microphone stream...")
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error("[AUDIO] Error while closing microphone stream: %s", e)
            finally:
                self._stream = None
        logger.info("[AUDIO] Microphone stream stopped.")

    def close(self) -> None:
        """Release microphone resources."""
        self.stop()

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
