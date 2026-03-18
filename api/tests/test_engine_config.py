"""Tests for engine configuration."""
import pytest


def test_default_engine_is_kokoro():
    """Default engine should be kokoro for backwards compatibility."""
    from api.src.core.config import Settings
    s = Settings()
    assert s.tts_engine == "kokoro"


def test_engine_from_env(monkeypatch):
    """Engine should be configurable via environment variable."""
    monkeypatch.setenv("TTS_ENGINE", "qwen3")
    from api.src.core.config import Settings
    s = Settings()
    assert s.tts_engine == "qwen3"


def test_invalid_engine_rejected(monkeypatch):
    """Only known engine names should be accepted."""
    monkeypatch.setenv("TTS_ENGINE", "invalid_engine")
    from api.src.core.config import Settings
    with pytest.raises(Exception):  # Pydantic ValidationError
        Settings()
