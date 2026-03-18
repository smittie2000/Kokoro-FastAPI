"""Integration tests for Qwen3-TTS — requires GPU and model download.

Run with: pytest api/tests/test_qwen3_integration.py -v -m gpu
Skip in CI: these tests are marked with @pytest.mark.gpu
"""
import os

import numpy as np
import pytest
import pytest_asyncio

# Skip entire module if no GPU or model not available
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("TTS_ENGINE") != "qwen3",
        reason="TTS_ENGINE != qwen3 — skipping integration tests"
    ),
]


@pytest_asyncio.fixture
async def qwen3_backend():
    """Load Qwen3-TTS backend with real model."""
    from api.src.inference.qwen3_tts import Qwen3TTS
    backend = Qwen3TTS(voices_dir="voices/qwen3")
    await backend.load_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    yield backend
    backend.unload()


@pytest.mark.asyncio
async def test_streaming_produces_audio(qwen3_backend):
    """Full streaming generation should yield audio chunks."""
    chunks = []
    async for chunk in qwen3_backend.generate(
        "Hello, this is a streaming test.",
        "test_voice",
        lang_code="en",
    ):
        assert chunk.audio.dtype == np.int16
        assert len(chunk.audio) > 0
        chunks.append(chunk)

    assert len(chunks) > 0
    total_samples = sum(len(c.audio) for c in chunks)
    duration_s = total_samples / 24000
    assert duration_s > 0.5, f"Audio too short: {duration_s:.2f}s"
    assert duration_s < 30, f"Audio too long: {duration_s:.2f}s"


@pytest.mark.asyncio
async def test_voice_prompt_caching(qwen3_backend):
    """Second generation with same voice should use cached prompt."""
    # First generation — creates voice prompt
    async for _ in qwen3_backend.generate("First call.", "test_voice"):
        break  # Just need one chunk

    assert "test_voice" in qwen3_backend._voice_prompts

    # Second generation — should use cache (no re-computation)
    prompt_before = id(qwen3_backend._voice_prompts["test_voice"])
    async for _ in qwen3_backend.generate("Second call.", "test_voice"):
        break

    prompt_after = id(qwen3_backend._voice_prompts["test_voice"])
    assert prompt_before == prompt_after, "Voice prompt was recomputed instead of cached"
