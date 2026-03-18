"""Tests for engine selection in ModelManager."""
import pytest
from unittest.mock import patch

from api.src.core.config import settings


@pytest.mark.asyncio
async def test_initialize_defaults_to_kokoro(monkeypatch):
    """With default config, ModelManager should create KokoroV1 backend."""
    monkeypatch.setattr(settings, "tts_engine", "kokoro")
    from api.src.inference.model_manager import ModelManager
    ModelManager._instance = None

    mgr = ModelManager()
    await mgr.initialize()

    from api.src.inference.kokoro_v1 import KokoroV1
    assert isinstance(mgr.get_backend(), KokoroV1)


@pytest.mark.asyncio
async def test_initialize_qwen3_creates_qwen3_backend(monkeypatch):
    """With TTS_ENGINE=qwen3, ModelManager should create Qwen3TTS backend."""
    monkeypatch.setattr(settings, "tts_engine", "qwen3")
    from api.src.inference.model_manager import ModelManager
    ModelManager._instance = None

    mgr = ModelManager()
    await mgr.initialize()

    from api.src.inference.qwen3_tts import Qwen3TTS
    assert isinstance(mgr.get_backend(), Qwen3TTS)


@pytest.mark.asyncio
async def test_current_backend_reflects_engine(monkeypatch):
    """current_backend property should return engine name."""
    monkeypatch.setattr(settings, "tts_engine", "qwen3")
    from api.src.inference.model_manager import ModelManager
    ModelManager._instance = None

    mgr = ModelManager()
    await mgr.initialize()
    assert mgr.current_backend == "qwen3"
