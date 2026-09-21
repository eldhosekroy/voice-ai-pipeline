"""
Language Model (LLM) component interface and Ollama Qwen implementation.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract interface for Large Language Models."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a text response given a prompt.

        :param prompt: User prompt text
        :return: Generated response text
        """
        pass


class QwenLLM(BaseLLM):
    """Qwen LLM via Ollama implementation stub."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5",
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name

        # TODO: Initialize Ollama API client connection
        logger.info(
            "[LLM] Initialized QwenLLM stub (model=%s, url=%s)",
            self.model_name,
            self.base_url,
        )

    def generate_response(self, prompt: str) -> str:
        """
        Send prompt to Qwen running locally via Ollama and return text response.
        """
        # TODO: Implement Ollama API call with Qwen model
        # 1. Prepare request payload
        # 2. Post request to Ollama endpoint
        # 3. Parse and return text response
        logger.info("[LLM] Sending request to Qwen...")
        return ""
