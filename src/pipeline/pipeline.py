"""
Main Voice AI Pipeline orchestrator.
Connects microphone input, VAD, STT, LLM, TTS, and speaker output.
"""

import logging
import time
from config.config import PipelineConfig
from src.audio.microphone import BaseAudioInput, Microphone
from src.vad.silero_vad import BaseVAD, SileroVAD
from src.stt.whisper import BaseSTT, WhisperSTT
from src.llm.qwen import BaseLLM, QwenLLM
from src.tts.kokoro import BaseTTS, KokoroTTS

logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Modular Voice AI Pipeline orchestrator.
    Employs dependency injection to allow independent component testing and replacement.
    """

    def __init__(
        self,
        config: PipelineConfig,
        audio_input: BaseAudioInput | None = None,
        vad: BaseVAD | None = None,
        stt: BaseSTT | None = None,
        llm: BaseLLM | None = None,
        tts: BaseTTS | None = None,
    ) -> None:
        self.config = config

        # Instantiate microphone audio input
        self.audio_input = audio_input or Microphone(
            sample_rate=config.sample_rate,
            chunk_size=512,
            channels=1,
            dtype="float32",
        )

        # Instantiate downstream modular component stubs
        self.vad = vad or SileroVAD(
            threshold=config.vad_threshold,
            min_speech_duration=config.vad_min_speech_duration,
            min_silence_duration=config.vad_min_silence_duration,
            sample_rate=config.sample_rate,
        )

        self.stt = stt or WhisperSTT(
            model_name=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

        self.llm = llm or QwenLLM(
            base_url=config.ollama_base_url,
            model_name=config.ollama_model,
        )

        self.tts = tts or KokoroTTS(
            voice=config.kokoro_voice,
            sample_rate=config.sample_rate,
        )

        logger.info("[Pipeline] VoicePipeline initialized successfully.")

    def run_microphone_test(self, duration_seconds: float = 2.0) -> bool:
        """
        Demonstrate microphone audio capture initialization, chunk reading, and shutdown (STEP 2 test).

        :param duration_seconds: Number of seconds to test microphone recording.
        :return: True if audio chunks were successfully read, False otherwise.
        """
        logger.info("[Pipeline] Testing microphone audio capture for %.1f seconds...", duration_seconds)
        chunks_read = 0
        samples_read = 0

        try:
            self.audio_input.start()
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                chunk = self.audio_input.read_chunk(timeout=0.5)
                if chunk is not None:
                    chunks_read += 1
                    samples_read += len(chunk)

            logger.info(
                "[AUDIO] Capture test complete: Read %d chunks (%d total samples).",
                chunks_read,
                samples_read,
            )
            return chunks_read > 0
        except Exception as e:
            logger.error("[AUDIO] Error during microphone capture test: %s", e)
            return False
        finally:
            self.audio_input.stop()

    def run(self) -> None:
        """Execute the voice pipeline main loop."""
        logger.info("[Pipeline] Starting voice pipeline...")
        # Currently demonstrating STEP 2 microphone capture
        self.run_microphone_test(duration_seconds=2.0)
