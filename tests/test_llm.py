"""
Unit tests for Qwen LLM module.
"""

import unittest
from src.llm.qwen import QwenLLM


class TestQwenLLM(unittest.TestCase):
    """Test Qwen LLM Ollama interface."""

    def test_llm_init(self) -> None:
        llm = QwenLLM(base_url="http://localhost:11434", model_name="qwen2.5")
        self.assertEqual(llm.model_name, "qwen2.5")
        self.assertTrue(isinstance(llm.is_available, bool))

    def test_empty_prompt(self) -> None:
        llm = QwenLLM()
        response = llm.generate_response("")
        self.assertEqual(response, "")


if __name__ == "__main__":
    unittest.main()
