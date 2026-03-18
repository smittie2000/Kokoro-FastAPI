"""Tests for Qwen3TTS backend.

These tests verify the backend interface without requiring the actual model.
Model loading and generation are mocked — integration tests with real model
are separate (require GPU + ~8GB VRAM + model download).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from api.src.inference.base import AudioChunk, BaseModelBackend


def test_qwen3_tts_inherits_base_backend():
    """Qwen3TTS must implement BaseModelBackend."""
    from api.src.inference.qwen3_tts import Qwen3TTS
    assert issubclass(Qwen3TTS, BaseModelBackend)


def test_qwen3_tts_not_loaded_initially():
    """Backend should not be loaded before load_model is called."""
    from api.src.inference.qwen3_tts import Qwen3TTS
    backend = Qwen3TTS(voices_dir="/tmp/test_voices")
    assert backend.is_loaded is False


def test_qwen3_tts_device_default():
    """Default device should come from settings."""
    from api.src.inference.qwen3_tts import Qwen3TTS
    backend = Qwen3TTS(voices_dir="/tmp/test_voices")
    assert backend.device in ("cpu", "cuda", "mps")


@pytest.mark.asyncio
async def test_qwen3_tts_generate_raises_when_not_loaded():
    """generate() should raise RuntimeError if model not loaded."""
    from api.src.inference.qwen3_tts import Qwen3TTS
    backend = Qwen3TTS(voices_dir="/tmp/test_voices")
    with pytest.raises(RuntimeError, match="not loaded"):
        async for _ in backend.generate("test", "voice"):
            pass


@pytest.mark.asyncio
async def test_qwen3_tts_generate_yields_audio_chunks():
    """generate() should yield AudioChunk objects with int16 audio."""
    from api.src.inference.qwen3_tts import Qwen3TTS

    backend = Qwen3TTS(voices_dir="/tmp/test_voices")

    # Mock the model
    mock_model = MagicMock()

    # Simulate streaming: 3 chunks of float32 PCM at 24kHz
    def fake_stream(**kwargs):
        for i in range(3):
            chunk = np.random.randn(2400).astype(np.float32) * 0.5
            yield chunk, 24000

    mock_model.stream_generate_voice_clone = MagicMock(side_effect=fake_stream)
    backend._model = mock_model

    # Patch voice prompt resolution to avoid voice file I/O
    backend._get_or_create_voice_prompt = MagicMock(return_value=MagicMock())

    chunks = []
    async for chunk in backend.generate("Hello world", "test_voice"):
        assert isinstance(chunk, AudioChunk)
        assert chunk.audio.dtype == np.int16
        assert len(chunk.audio) > 0
        chunks.append(chunk)

    assert len(chunks) == 3
    # Verify voice prompt was resolved
    backend._get_or_create_voice_prompt.assert_called_once_with("test_voice")


@pytest.mark.asyncio
async def test_qwen3_tts_unload_clears_model():
    """unload() should free model and set is_loaded to False."""
    from api.src.inference.qwen3_tts import Qwen3TTS
    backend = Qwen3TTS(voices_dir="/tmp/test_voices")
    backend._model = MagicMock()  # Simulate loaded model
    assert backend.is_loaded is True
    backend.unload()
    assert backend.is_loaded is False
