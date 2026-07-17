from __future__ import annotations

import base64
import os
import time
from typing import List

import requests
from dotenv import load_dotenv

from ..reporting import detail_print
from .base import BaseVLM
from .openai_compatible import (
    bearer_headers,
    post_with_proxy_fallback,
    resolve_openai_compatible_config,
)


load_dotenv()

DEFAULT_EVAL_VLM_MODEL = "google/gemini-2.5-flash"
MODEL_ALIASES = {
    "gemini-flash": "google/gemini-3.1-flash-image-preview",
    "gemini-2.5": "google/gemini-2.5-flash-image",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt5-image": "openai/gpt-5-image",
    "gpt5-image-mini": "openai/gpt-5-image-mini",
}
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleVisionLanguageModel(BaseVLM):
    """VLM wrapper for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        model_name: str,
        timeout: int = 60,
        max_tokens: int = 16,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ):
        super().__init__(model_name=model_name)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.api_key = None
        self.api_base = None

    def load_model(self):
        config = resolve_openai_compatible_config()
        self.api_key, self.api_base = config.api_key, config.api_base
        if not self.api_key:
            raise ValueError(
                "No OpenAI-compatible API key found. Set PROVISE_API_KEY, "
                "OPENAI_API_KEY, or OPENROUTER_API_KEY."
            )
        if not self.api_base:
            raise ValueError(
                "No OpenAI-compatible API base found. Set PROVISE_API_BASE or "
                "OPENAI_BASE_URL."
            )
        self.api_base = self.api_base.rstrip("/")
        detail_print(f"Eval VLM initialized: {self.model_name}")

    def predict(self, image_path: str, question: str) -> str:
        return self.predict_multi([image_path], question)

    def predict_multi(self, image_paths: List[str], question: str) -> str:
        if self.api_key is None:
            self.load_model()

        content = [{"type": "text", "text": question}]
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"image does not exist: {image_path}")
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{_mime_type(image_path)};base64,{image_b64}"},
                }
            )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post_with_proxy_fallback(payload)
            except requests.exceptions.RequestException as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        "VLM request failed after "
                        f"{attempt + 1} attempt(s): {type(exc).__name__}: {exc}"
                    ) from exc
                self._wait_before_retry(type(exc).__name__, attempt)
                continue

            if response.status_code == 200:
                try:
                    return response.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            "VLM API returned a malformed response after "
                            f"{attempt + 1} attempt(s): {type(exc).__name__}: {exc}"
                        ) from exc
                    self._wait_before_retry("malformed response", attempt)
                    continue

            if response.status_code not in RETRYABLE_STATUS_CODES or attempt == self.max_retries:
                raise RuntimeError(
                    f"VLM API error {response.status_code}: {response.text[:200]}"
                )
            self._wait_before_retry(f"HTTP {response.status_code}", attempt)

        raise RuntimeError("VLM request exhausted retries")

    def _wait_before_retry(self, reason: str, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        detail_print(
            f"Eval VLM request failed with {reason}; "
            f"retrying in {delay:g}s ({attempt + 1}/{self.max_retries})..."
        )
        time.sleep(delay)

    def _post_with_proxy_fallback(self, payload: dict):
        return post_with_proxy_fallback(
            f"{self.api_base}/chat/completions",
            headers=bearer_headers(self.api_key),
            json=payload,
            timeout=self.timeout,
            on_proxy_fallback=lambda: detail_print(
                "Eval VLM proxy unavailable, retrying without proxy..."
            ),
        )


def create_eval_vlm(
    timeout: int = 60,
    max_tokens: int | None = None,
    model_name: str | None = None,
) -> OpenAICompatibleVisionLanguageModel:
    model_name = (
        model_name
        or os.getenv("PROVISE_PARSER_MODEL")
        or DEFAULT_EVAL_VLM_MODEL
    ).strip()
    model_name = MODEL_ALIASES.get(model_name, model_name)
    if max_tokens is None:
        max_tokens = int(os.getenv("PROVISE_PARSER_MAX_TOKENS", "512"))
    max_retries = int(os.getenv("PROVISE_PARSER_MAX_RETRIES", "2"))
    retry_backoff = float(os.getenv("PROVISE_PARSER_RETRY_BACKOFF", "1"))
    return OpenAICompatibleVisionLanguageModel(
        model_name=model_name or DEFAULT_EVAL_VLM_MODEL,
        timeout=timeout,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


def _mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"
