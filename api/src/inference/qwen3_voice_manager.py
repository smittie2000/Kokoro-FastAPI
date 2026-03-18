"""Voice manager for Qwen3-TTS zero-shot voice cloning.

Qwen3-TTS uses reference audio + transcript pairs for voice cloning,
not pre-baked .pt tensor files like Kokoro. Each voice profile is a
directory containing ref.wav and ref.json.

Directory structure:
    voices/qwen3/
        my_voice/
            ref.wav      # 10-30s reference audio
            ref.json     # {"ref_text": "transcript of the reference audio"}
"""

import json
import os
from typing import List, Tuple

from loguru import logger


class Qwen3VoiceManager:
    """Manages Qwen3 voice profiles (reference audio + text pairs)."""

    def __init__(self, voices_dir: str):
        self._voices_dir = voices_dir

    def list_voices(self) -> List[str]:
        """List available voice profile names."""
        if not os.path.isdir(self._voices_dir):
            return []
        return sorted(
            name
            for name in os.listdir(self._voices_dir)
            if os.path.isdir(os.path.join(self._voices_dir, name))
            and os.path.exists(os.path.join(self._voices_dir, name, "ref.wav"))
        )

    def get_voice_reference(self, voice_name: str) -> Tuple[str, str]:
        """Get reference audio path and transcript for a voice.

        Args:
            voice_name: Name of the voice profile directory

        Returns:
            Tuple of (path_to_ref_wav, reference_text)

        Raises:
            FileNotFoundError: If voice directory, ref.wav, or ref.json missing
        """
        voice_dir = os.path.join(self._voices_dir, voice_name)
        if not os.path.isdir(voice_dir):
            raise FileNotFoundError(f"Voice profile not found: {voice_name}")

        wav_path = os.path.join(voice_dir, "ref.wav")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"ref.wav not found in voice profile: {voice_name}")

        json_path = os.path.join(voice_dir, "ref.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"ref.json not found in voice profile: {voice_name}")

        with open(json_path, "r") as f:
            data = json.load(f)

        ref_text = data.get("ref_text", "")
        if not ref_text:
            logger.warning(f"Empty ref_text in voice profile: {voice_name}")

        return wav_path, ref_text
