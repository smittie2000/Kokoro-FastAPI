"""Tests for Qwen3 voice manager."""
import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def voice_dir(tmp_path):
    """Create a temporary voice directory with a test voice."""
    voice_path = tmp_path / "test_voice"
    voice_path.mkdir()

    # Create a short reference WAV (1 second of silence at 24kHz)
    audio = np.zeros(24000, dtype=np.float32)
    sf.write(str(voice_path / "ref.wav"), audio, 24000)

    # Create reference text JSON
    ref_data = {"ref_text": "This is the reference transcript."}
    with open(voice_path / "ref.json", "w") as f:
        json.dump(ref_data, f)

    return tmp_path


def test_list_voices(voice_dir):
    from api.src.inference.qwen3_voice_manager import Qwen3VoiceManager
    mgr = Qwen3VoiceManager(str(voice_dir))
    voices = mgr.list_voices()
    assert "test_voice" in voices


def test_list_voices_empty(tmp_path):
    from api.src.inference.qwen3_voice_manager import Qwen3VoiceManager
    mgr = Qwen3VoiceManager(str(tmp_path))
    voices = mgr.list_voices()
    assert voices == []


def test_get_voice_reference(voice_dir):
    from api.src.inference.qwen3_voice_manager import Qwen3VoiceManager
    mgr = Qwen3VoiceManager(str(voice_dir))
    ref_audio_path, ref_text = mgr.get_voice_reference("test_voice")
    assert os.path.exists(ref_audio_path)
    assert ref_audio_path.endswith("ref.wav")
    assert ref_text == "This is the reference transcript."


def test_get_voice_reference_missing(voice_dir):
    from api.src.inference.qwen3_voice_manager import Qwen3VoiceManager
    mgr = Qwen3VoiceManager(str(voice_dir))
    with pytest.raises(FileNotFoundError):
        mgr.get_voice_reference("nonexistent_voice")


def test_get_voice_reference_no_json(voice_dir):
    """Voice directory with WAV but no JSON should raise."""
    bad_voice = voice_dir / "bad_voice"
    bad_voice.mkdir()
    audio = np.zeros(24000, dtype=np.float32)
    sf.write(str(bad_voice / "ref.wav"), audio, 24000)

    from api.src.inference.qwen3_voice_manager import Qwen3VoiceManager
    mgr = Qwen3VoiceManager(str(voice_dir))
    with pytest.raises(FileNotFoundError, match="ref.json"):
        mgr.get_voice_reference("bad_voice")
