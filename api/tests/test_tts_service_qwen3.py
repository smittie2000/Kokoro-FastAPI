"""Tests for TTSService with Qwen3 engine.

Verifies that TTSService correctly routes to Qwen3TTS backend
without Kokoro-specific phonemization or voice resolution.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from api.src.core.config import settings
from api.src.inference.base import AudioChunk


@pytest.mark.asyncio
async def test_process_chunk_uses_text_branch_for_qwen3(monkeypatch):
    """_process_chunk should use text-based generation for Qwen3TTS, not legacy token branch."""
    monkeypatch.setattr(settings, "tts_engine", "qwen3")

    from api.src.inference.qwen3_tts import Qwen3TTS
    from api.src.services.tts_service import TTSService
    from api.src.services.streaming_audio_writer import StreamingAudioWriter

    service = TTSService()
    mock_manager = MagicMock()

    # Create a mock Qwen3TTS backend
    mock_backend = MagicMock(spec=Qwen3TTS)

    # generate() should yield AudioChunks
    async def fake_generate(*args, **kwargs):
        yield AudioChunk(audio=np.zeros(2400, dtype=np.int16))

    mock_manager.get_backend.return_value = mock_backend
    mock_manager.generate = fake_generate
    service.model_manager = mock_manager

    writer = MagicMock(spec=StreamingAudioWriter)
    writer.write_chunk.return_value = b"\x00" * 100

    chunks = []
    async for chunk in service._process_chunk(
        chunk_text="Hello world",
        tokens=[],  # Empty tokens — Qwen3 uses text, not tokens
        voice_name="test_voice",
        voice_path="voices/qwen3/test_voice",
        speed=1.0,
        writer=writer,
        output_format=None,  # Raw audio mode
        lang_code="en",
    ):
        chunks.append(chunk)

    # Should have generated audio (not fallen into legacy branch which would fail)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_list_voices_returns_qwen3_voices(monkeypatch, tmp_path):
    """list_voices() should return Qwen3 voice profiles when engine is qwen3."""
    monkeypatch.setattr(settings, "tts_engine", "qwen3")

    # Create a fake voice profile
    voice_dir = tmp_path / "test_voice"
    voice_dir.mkdir()
    sf.write(str(voice_dir / "ref.wav"), np.zeros(24000, dtype=np.float32), 24000)
    with open(voice_dir / "ref.json", "w") as f:
        json.dump({"ref_text": "test"}, f)

    from api.src.services.tts_service import TTSService
    from api.src.inference.qwen3_voice_manager import Qwen3VoiceManager

    service = TTSService()
    service.model_manager = MagicMock()
    service._qwen3_voice_manager = Qwen3VoiceManager(str(tmp_path))

    voices = await service.list_voices()
    assert "test_voice" in voices
