"""
Main Voice AI Pipeline orchestrator.
Connects Microphone -> Silero VAD -> Whisper STT -> Qwen via Ollama -> Kokoro TTS -> Speaker.
"""

import logging
import signal
import sys
import time
import numpy as np

from config.config import PipelineConfig
from src.audio.microphone import BaseAudioInput, Microphone
from src.audio.output import BaseAudioOutput, Speaker
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
        speaker: BaseAudioOutput | None = None,
    ) -> None:
        self.config = config

        # Component 1: Microphone audio input
        self.audio_input = audio_input or Microphone(
            sample_rate=config.sample_rate,
            chunk_size=512,
            channels=1,
            dtype="float32",
        )

        # Component 2: Silero VAD
        self.vad = vad or SileroVAD(
            threshold=config.vad_threshold,
            min_speech_duration=config.vad_min_speech_duration,
            min_silence_duration=config.vad_min_silence_duration,
            sample_rate=config.sample_rate,
            device=config.vad_device,
        )

        # Component 3: Whisper STT
        self.stt = stt or WhisperSTT(
            model_name=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

        # Component 4: Qwen via Ollama LLM
        self.llm = llm or QwenLLM(
            base_url=config.ollama_base_url,
            model_name=config.ollama_model,
        )

        # Component 5: Kokoro TTS
        self.tts = tts or KokoroTTS(
            voice=config.kokoro_voice,
            sample_rate=config.sample_rate,
            device=config.kokoro_device,
        )

        # Component 6: Speaker audio output
        self.speaker = speaker or Speaker()

        self.is_running: bool = False
        logger.info("[Pipeline] VoicePipeline initialized successfully.")

    def process_once(self) -> bool:
        """
        Process a single audio chunk through the VAD and downstream pipeline.

        :return: True if a speech response loop was executed, False otherwise.
        """
        chunk = self.audio_input.read_chunk(timeout=0.2)
        if chunk is None or len(chunk) == 0:
            return False

        # 1. Silero VAD processing
        speech_segment = self.vad.process_chunk(chunk)
        if speech_segment is None:
            return False

        # 2. Speech segment detected -> Whisper STT transcription
        transcription = self.stt.transcribe(speech_segment)
        if not transcription or not transcription.strip():
            logger.info("[Pipeline] Silence or empty transcription. Returning to listening...")
            return False

        # 3. LLM response generation via Ollama (Qwen)
        ai_response = self.llm.generate_response(transcription)
        if not ai_response or not ai_response.strip():
            logger.warning("[Pipeline] Empty response received from LLM.")
            return False

        # 4. Kokoro TTS speech audio synthesis
        audio_data, sr = self.tts.synthesize(ai_response)
        if audio_data is None or len(audio_data) == 0:
            logger.warning("[Pipeline] Failed to generate TTS audio.")
            return False

        # 5. Speaker audio playback
        self.speaker.play(audio_data, sample_rate=sr)

        logger.info("[Pipeline] Turn complete. Returning to listening state...")
        return True

    def run_microphone_test(self, duration_seconds: float = 2.0) -> bool:
        """Run short microphone test."""
        logger.info("[Pipeline] Testing microphone audio capture for %.1f seconds...", duration_seconds)
        chunks_read = 0
        try:
            self.audio_input.start()
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                chunk = self.audio_input.read_chunk(timeout=0.5)
                if chunk is not None:
                    chunks_read += 1
            logger.info("[AUDIO] Capture test complete: Read %d chunks.", chunks_read)
            return chunks_read > 0
        finally:
            self.audio_input.stop()

    def run(self) -> None:
        """
        Execute continuous real-time voice pipeline loop.
        Press Ctrl+C to stop.
        """
        logger.info("[Pipeline] Starting real-time Voice AI Pipeline...")
        logger.info("[Pipeline] Listening for user speech (Press Ctrl+C to exit)...")

        self.is_running = True
        try:
            self.audio_input.start()
            while self.is_running:
                self.process_once()
        except KeyboardInterrupt:
            logger.info("\n[Pipeline] KeyboardInterrupt received. Stopping pipeline...")
        except Exception as e:
            logger.error("[Pipeline] Pipeline execution error: %s", e)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all running pipeline components."""
        if not self.is_running:
            return
        logger.info("[Pipeline] Shutting down Voice AI Pipeline...")
        self.is_running = False
        try:
            self.audio_input.stop()
        except Exception as e:
            logger.error("[Pipeline] Error stopping audio input: %s", e)
        try:
            self.speaker.stop()
        except Exception as e:
            logger.error("[Pipeline] Error stopping speaker: %s", e)
        logger.info("[Pipeline] Voice AI Pipeline shutdown complete.")
