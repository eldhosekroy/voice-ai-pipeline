"""
Unit tests for Silero VAD module.
"""

import unittest
import numpy as np
from src.vad.silero_vad import SileroVAD


class TestSileroVAD(unittest.TestCase):
    """Test Silero VAD speech detection logic."""

    def setUp(self) -> None:
        self.vad = SileroVAD(threshold=0.5, sample_rate=16000)

    def test_silence_chunk(self) -> None:
        # 512 zeros (silence) should return False for speech detection
        silence_chunk = np.zeros(512, dtype=np.float32)
        is_speech = self.vad.is_speech(silence_chunk)
        self.assertFalse(is_speech)

    def test_vad_process_chunk(self) -> None:
        # Processing silent chunk should return None
        silence_chunk = np.zeros(512, dtype=np.float32)
        segment = self.vad.process_chunk(silence_chunk)
        self.assertIsNone(segment)

    def test_vad_reset(self) -> None:
        self.vad.reset()
        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(len(self.vad.speech_chunks), 0)


if __name__ == "__main__":
    unittest.main()
