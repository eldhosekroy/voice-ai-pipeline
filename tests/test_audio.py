"""
Unit tests for Microphone input and Speaker output modules.
"""

import unittest
import numpy as np
from src.audio.microphone import Microphone
from src.audio.output import Speaker


class TestAudioModules(unittest.TestCase):
    """Test audio capture and playback module abstractions."""

    def test_microphone_init(self) -> None:
        mic = Microphone(sample_rate=16000, chunk_size=512)
        self.assertEqual(mic.sample_rate, 16000)
        self.assertEqual(mic.chunk_size, 512)
        self.assertFalse(mic._is_running)

    def test_speaker_init(self) -> None:
        speaker = Speaker()
        self.assertFalse(speaker._is_playing)

    def test_speaker_empty_play(self) -> None:
        speaker = Speaker()
        # Playing empty array should handle gracefully without exception
        speaker.play(np.array([], dtype=np.float32), 16000)


if __name__ == "__main__":
    unittest.main()
