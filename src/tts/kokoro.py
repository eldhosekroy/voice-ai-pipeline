"""
Text-to-Speech (TTS) component using genuine Kokoro ONNX model.
Synthesizes response text into floating-point audio samples.
"""

from abc import ABC, abstractmethod
import logging
import os
import requests
import numpy as np

try:
    import kokoro_onnx
except ImportError:
    kokoro_onnx = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

logger = logging.getLogger(__name__)

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

KOKORO_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "kokoro")
KOKORO_MODEL_PATH = os.path.join(KOKORO_CACHE_DIR, "kokoro-v1.0.int8.onnx")
KOKORO_VOICES_PATH = os.path.join(KOKORO_CACHE_DIR, "voices-v1.0.bin")


class BaseTTS(ABC):
    """Abstract interface for Text-to-Speech synthesis engines."""

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        """
        Synthesize input text into audio samples and sample rate.

        :param text: Input text string
        :return: Tuple of (1D float32 numpy audio array, sample_rate_hz)
        """
        pass


class KokoroTTS(BaseTTS):
    """
    Genuine Kokoro Text-to-Speech implementation using kokoro-onnx.
    Converts text responses into floating-point audio waveforms using Kokoro model weights.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        sample_rate: int = 24000,
        device: str = "cpu",
    ) -> None:
        self.voice = voice
        self.sample_rate = sample_rate
        self.device = device

        self.kokoro_engine = None
        self.is_genuine_kokoro: bool = False

        self._init_kokoro_model()

    def _download_file(self, url: str, target_path: str) -> None:
        """Download file with stream progress logging."""
        logger.info("[TTS] Downloading Kokoro model asset: %s...", url)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        response = requests.get(url, allow_redirects=True, stream=True, timeout=60.0)
        response.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)
        logger.info("[TTS] Successfully downloaded %s.", target_path)

    def _init_kokoro_model(self) -> None:
        """Load or download genuine Kokoro ONNX model and voice weights."""
        if kokoro_onnx is None:
            logger.error("[TTS] kokoro-onnx package is not installed. Run `pip install kokoro-onnx`.")
            self.is_genuine_kokoro = False
            return

        try:
            if not os.path.exists(KOKORO_MODEL_PATH) or os.path.getsize(KOKORO_MODEL_PATH) < 10000000:
                self._download_file(KOKORO_MODEL_URL, KOKORO_MODEL_PATH)

            if not os.path.exists(KOKORO_VOICES_PATH) or os.path.getsize(KOKORO_VOICES_PATH) < 10000000:
                self._download_file(KOKORO_VOICES_URL, KOKORO_VOICES_PATH)

            logger.info(
                "[TTS] Initializing Genuine Kokoro ONNX TTS model (voice=%s, device=%s)...",
                self.voice,
                self.device,
            )
            self.kokoro_engine = kokoro_onnx.Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
            self.is_genuine_kokoro = True
            logger.info("[TTS] Genuine Kokoro TTS engine initialized successfully.")
        except Exception as e:
            logger.error("[TTS] Failed to initialize Genuine Kokoro TTS model: %s", e)
            self.kokoro_engine = None
            self.is_genuine_kokoro = False

    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        """
        Synthesize input text into audio using genuine Kokoro TTS model.

        :param text: Text string to synthesize
        :return: Tuple of (audio_samples_array, sample_rate)
        """
        if not text or not text.strip():
            return None, self.sample_rate

        # Primary Implementation: Genuine Kokoro ONNX
        if self.is_genuine_kokoro and self.kokoro_engine is not None:
            try:
                logger.info("[TTS] Generating speech using Genuine Kokoro TTS (voice=%s)...", self.voice)
                audio, sr = self.kokoro_engine.create(text, voice=self.voice)
                logger.info("[TTS] Audio generated (%d samples, %d Hz).", len(audio), sr)
                return audio.astype(np.float32), sr
            except Exception as e:
                logger.error("[TTS] Genuine Kokoro synthesis error: %s", e)

        # Explicit Warning when falling back
        logger.warning(
            "[TTS] [FALLBACK WARNING] Genuine Kokoro model is unavailable. Falling back to pyttsx3. This is NOT Kokoro output."
        )

        if pyttsx3 is not None:
            try:
                import tempfile
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()

                if sf is not None and os.path.exists(tmp_path):
                    data, sr = sf.read(tmp_path, dtype="float32")
                    os.remove(tmp_path)
                    if len(data.shape) > 1:
                        data = data[:, 0]
                    return data, sr
            except Exception as fallback_err:
                logger.error("[TTS] Fallback pyttsx3 synthesis failed: %s", fallback_err)

        return None, self.sample_rate
