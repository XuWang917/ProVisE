from provise.models.openai_compatible import (
    DEFAULT_OPENAI_API_BASE,
    DEFAULT_OPENROUTER_API_BASE,
    resolve_openai_compatible_config,
)


def _clear_provider_env(monkeypatch):
    for name in (
        "PROVISE_API_PROVIDER",
        "PROVISE_API_KEY",
        "PROVISE_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_BASE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_provise_provider_config_takes_precedence(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PROVISE_API_KEY", "proxy-key")
    monkeypatch.setenv("PROVISE_API_BASE", "https://proxy.example/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    config = resolve_openai_compatible_config()

    assert config.api_key == "proxy-key"
    assert config.api_base == "https://proxy.example/v1/"
    assert config.source == "provise"


def test_standard_openai_environment_is_supported(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    config = resolve_openai_compatible_config()

    assert config.api_key == "openai-key"
    assert config.api_base == DEFAULT_OPENAI_API_BASE
    assert config.source == "openai"


def test_custom_provider_requires_an_explicit_base(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PROVISE_API_KEY", "proxy-key")

    config = resolve_openai_compatible_config()

    assert config.api_key == "proxy-key"
    assert config.api_base == ""


def test_openrouter_environment_uses_openrouter_default(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    config = resolve_openai_compatible_config()

    assert config.api_key == "router-key"
    assert config.api_base == DEFAULT_OPENROUTER_API_BASE
    assert config.source == "openrouter"


def test_explicit_openrouter_provider_overrides_proxy_config(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PROVISE_API_PROVIDER", "openrouter")
    monkeypatch.setenv("PROVISE_API_KEY", "proxy-key")
    monkeypatch.setenv("PROVISE_API_BASE", "https://proxy.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    config = resolve_openai_compatible_config()

    assert config.api_key == "router-key"
    assert config.api_base == DEFAULT_OPENROUTER_API_BASE
    assert config.source == "openrouter"
