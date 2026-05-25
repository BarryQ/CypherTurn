"""Unified LLM API client for evaluation.

Supports any OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.).
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def call(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Tuple[Optional[str], int, int]:
        """Call the LLM and return (response_text, input_tokens, output_tokens)."""
        ...


class OpenAIClient(BaseLLMClient):
    """Client for any OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, base_url: str, timeout: int = 120):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def call(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Tuple[Optional[str], int, int]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            usage = response.usage
            return content, usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None, 0, 0


def get_client(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> BaseLLMClient:
    """Create an OpenAI-compatible LLM client from environment or explicit args."""
    return OpenAIClient(
        model=model,
        api_key=api_key or os.environ.get("LLM_API_KEY", ""),
        base_url=base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        timeout=timeout,
    )
