"""Shared configuration and transport for OpenAI-compatible APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import requests


DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str | None
    api_base: str
    source: str


def resolve_openai_compatible_config(
    provider_override: str | None = None,
) -> OpenAICompatibleConfig:
    """Resolve one provider-neutral OpenAI-compatible API configuration."""
    provider = (
        provider_override
        or _env("PROVISE_API_PROVIDER")
        or ""
    ).lower()
    if provider == "openrouter":
        return OpenAICompatibleConfig(
            api_key=_env("OPENROUTER_API_KEY"),
            api_base=_env("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE,
            source="openrouter",
        )
    if provider == "openai":
        return OpenAICompatibleConfig(
            api_key=_env("OPENAI_API_KEY"),
            api_base=(
                _env("OPENAI_BASE_URL")
                or _env("OPENAI_API_BASE")
                or DEFAULT_OPENAI_API_BASE
            ),
            source="openai",
        )

    provise_key = _env("PROVISE_API_KEY")
    provise_base = _env("PROVISE_API_BASE")
    if provise_key or provise_base:
        return OpenAICompatibleConfig(
            api_key=provise_key,
            api_base=provise_base or "",
            source="provise",
        )

    openai_key = _env("OPENAI_API_KEY")
    openai_base = _env("OPENAI_BASE_URL") or _env("OPENAI_API_BASE")
    if openai_key or openai_base:
        return OpenAICompatibleConfig(
            api_key=openai_key,
            api_base=openai_base or DEFAULT_OPENAI_API_BASE,
            source="openai",
        )

    openrouter_key = _env("OPENROUTER_API_KEY")
    openrouter_base = _env("OPENROUTER_API_BASE")
    if openrouter_key or openrouter_base:
        return OpenAICompatibleConfig(
            api_key=openrouter_key,
            api_base=openrouter_base or DEFAULT_OPENROUTER_API_BASE,
            source="openrouter",
        )

    return OpenAICompatibleConfig(
        api_key=None,
        api_base=DEFAULT_OPENAI_API_BASE,
        source="default",
    )


def bearer_headers(api_key: str, *, json_content: bool = True) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def post_with_proxy_fallback(
    url: str,
    *,
    timeout: int | float,
    on_proxy_fallback: Callable[[], None] | None = None,
    **kwargs: Any,
) -> requests.Response:
    """POST using configured proxies, then retry directly on proxy failure."""
    proxies = proxy_config()
    try:
        return requests.post(
            url,
            timeout=timeout,
            proxies=proxies,
            **kwargs,
        )
    except requests.exceptions.ProxyError:
        if not proxies:
            raise
        if on_proxy_fallback is not None:
            on_proxy_fallback()
        with requests.Session() as session:
            session.trust_env = False
            return session.post(url, timeout=timeout, **kwargs)


def proxy_config() -> dict[str, str] | None:
    proxy_url = (
        _env("HTTPS_PROXY")
        or _env("https_proxy")
        or _env("HTTP_PROXY")
        or _env("http_proxy")
    )
    return {"http": proxy_url, "https": proxy_url} if proxy_url else None


def _env(name: str) -> str | None:
    value = str(os.getenv(name) or "").strip()
    return value or None
