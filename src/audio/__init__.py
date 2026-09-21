"""Audio package for microphone input and speaker output."""

from src.audio.microphone import BaseAudioInput, Microphone
from src.audio.output import BaseAudioOutput, Speaker

__all__ = ["BaseAudioInput", "Microphone", "BaseAudioOutput", "Speaker"]
