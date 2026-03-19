"""Qwen3-TTS streaming backend using dffdeeq/Qwen3-TTS-streaming patches.

This backend implements true model-level streaming: audio chunks are yielded
as the autoregressive model generates codec tokens, decoded in sliding windows.
Each chunk is ~333ms of audio (emit_every_frames=8 at 12Hz = 667ms, or 4 = 333ms).

Voice cloning is zero-shot: provide a reference WAV + transcript and the model
mimics that voice. No training, no per-voice model files.

Reference: https://github.com/dffdeeq/Qwen3-TTS-streaming
"""

import asyncio
from typing import AsyncGenerator, Dict, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger

from ..core.config import settings
from .base import AudioChunk, BaseModelBackend
from .qwen3_voice_manager import Qwen3VoiceManager


# Default streaming parameters
EMIT_EVERY_FRAMES = 8        # Emit chunk every 8 codec frames (~667ms at 12Hz)
DECODE_WINDOW_FRAMES = 80    # Sliding decode window size
MAX_FRAMES = 10000           # Safety limit (~14 minutes at 12Hz)


def _detect_flash_attn() -> bool:
    """Check if flash-attn package is installed and importable."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# Detect once at import time (avoids repeated import attempts)
FLASH_ATTN_AVAILABLE = _detect_flash_attn()


class Qwen3TTS(BaseModelBackend):
    """Qwen3-TTS backend with true model-level streaming."""

    def __init__(self, voices_dir: str):
        super().__init__()
        self._device = settings.get_device()
        self._model = None
        self._attn_implementation: str | None = None
        self._voice_prompts: Dict[str, object] = {}  # Cached voice prompts
        self._voice_manager = Qwen3VoiceManager(voices_dir)
        self._voices_dir = voices_dir

    async def load_model(self, path: str) -> None:
        """Load Qwen3-TTS model from HuggingFace or local path.

        Args:
            path: HuggingFace model ID or local directory
                  e.g. "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        """
        try:
            logger.info(f"Loading Qwen3-TTS model from {path} on {self._device}")

            def _load_sync():
                from qwen_tts import Qwen3TTSModel

                # Resolve dtype and device_map per backend
                if self._device == "cuda":
                    dtype = torch.bfloat16
                    device_map = "cuda"
                elif self._device == "mps":
                    dtype = torch.float32  # MPS has limited bf16 support
                    device_map = "mps"
                else:
                    dtype = torch.float32
                    device_map = None

                # Auto-detect best attention implementation:
                #   CUDA + flash-attn installed → flash_attention_2 (fastest)
                #   CUDA without flash-attn    → sdpa (PyTorch native, still good)
                #   MPS (Apple Silicon)        → sdpa (Metal-accelerated)
                #   CPU                        → None (eager default)
                attn_impl = None
                if self._device == "cuda" and FLASH_ATTN_AVAILABLE:
                    attn_impl = "flash_attention_2"
                    logger.info("FlashAttention-2 detected — enabling for all attention layers")
                elif self._device in ("cuda", "mps"):
                    attn_impl = "sdpa"
                    logger.info(f"Using PyTorch SDPA attention on {self._device}")

                model = Qwen3TTSModel.from_pretrained(
                    path,
                    device_map=device_map,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )

                # Enable streaming optimizations on CUDA (torch.compile + CUDA graphs)
                if self._device == "cuda":
                    logger.info("Enabling streaming optimizations (torch.compile + CUDA graphs)")
                    model.enable_streaming_optimizations(
                        decode_window_frames=DECODE_WINDOW_FRAMES,
                        use_compile=True,
                        use_cuda_graphs=True,
                        compile_mode="reduce-overhead",
                        compile_talker=True,
                        compile_codebook_predictor=True,
                    )

                return model, attn_impl

            self._model, self._attn_implementation = await asyncio.to_thread(_load_sync)
            logger.info(f"Qwen3-TTS model loaded on {self._device} (attn={self._attn_implementation or 'eager'})")

        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3-TTS model: {e}")

    def _get_or_create_voice_prompt(self, voice_name: str):
        """Get cached voice prompt or create from reference audio.

        Voice prompts are expensive to compute (~1s) but reusable across
        generations. Cache them in memory keyed by voice name.
        """
        if voice_name in self._voice_prompts:
            return self._voice_prompts[voice_name]

        ref_audio_path, ref_text = self._voice_manager.get_voice_reference(voice_name)

        logger.info(f"Creating voice prompt for '{voice_name}' from {ref_audio_path}")
        prompts = self._model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        prompt = prompts[0]

        self._voice_prompts[voice_name] = prompt
        logger.info(f"Voice prompt cached for '{voice_name}'")
        return prompt

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio with true model-level streaming.

        Yields AudioChunk objects as the model generates them — each chunk
        contains ~333-667ms of int16 audio at 24kHz.

        Args:
            text: Text to synthesize
            voice: Voice name (str) matching a profile in voices/qwen3/
                   or tuple of (name, path) for compatibility with Kokoro API
            speed: Speed multiplier (NOTE: Qwen3-TTS has no native speed control.
                   This parameter is accepted for ABC compatibility but ignored.
                   Post-processing resampling could be added later if needed.)
            lang_code: Language code (e.g. "en", "zh") — mapped to Qwen3 format
        """
        if not self.is_loaded:
            raise RuntimeError("Qwen3-TTS model not loaded")

        # Extract voice name from tuple if needed (Kokoro API compat)
        if isinstance(voice, tuple):
            voice_name = voice[0]
        else:
            voice_name = voice

        if speed != 1.0:
            logger.warning(f"Qwen3-TTS does not support speed={speed}, generating at 1.0x")

        # Map language codes
        language = self._map_language(lang_code)

        # Queue-based bridge: sync generator thread → async generator
        # This gives true streaming — chunks are yielded as they're generated,
        # not collected and returned all at once.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream_sync():
            """Run synchronous streaming in thread, push chunks to async queue."""
            try:
                voice_prompt = self._get_or_create_voice_prompt(voice_name)

                for pcm_chunk, sr in self._model.stream_generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_prompt,
                    emit_every_frames=EMIT_EVERY_FRAMES,
                    decode_window_frames=DECODE_WINDOW_FRAMES,
                    max_frames=MAX_FRAMES,
                    use_optimized_decode=True,
                ):
                    assert sr == 24000, f"Expected 24kHz sample rate, got {sr}Hz"
                    asyncio.run_coroutine_threadsafe(queue.put((pcm_chunk, sr)), loop).result()

                # Signal completion
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result()

        try:
            # Start generation in background thread
            thread_future = loop.run_in_executor(None, _stream_sync)

            # Yield chunks as they arrive from the queue
            while True:
                item = await queue.get()
                if item is None:
                    break  # Generation complete
                if isinstance(item, Exception):
                    raise item
                pcm_chunk, sr = item
                audio_int16 = np.clip(pcm_chunk * 32767, -32768, 32767).astype(np.int16)
                yield AudioChunk(audio=audio_int16)

            # Ensure thread finished cleanly
            await thread_future

        except Exception as e:
            logger.error(f"Qwen3-TTS generation failed: {e}")
            if self._device == "cuda":
                self._clear_memory()
            raise RuntimeError(f"Qwen3-TTS generation failed: {e}")

    def _map_language(self, lang_code: Optional[str]) -> str:
        """Map ISO 639-1 language codes to Qwen3-TTS language names."""
        if not lang_code:
            return "Auto"

        mapping = {
            "en": "English", "zh": "Chinese", "ja": "Japanese",
            "ko": "Korean", "fr": "French", "de": "German",
            "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
            "ar": "Arabic", "it": "Italian", "nl": "Dutch",
            "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
            "th": "Thai", "id": "Indonesian", "hi": "Hindi",
        }
        return mapping.get(lang_code.lower()[:2], "Auto")

    def _clear_memory(self) -> None:
        """Clear GPU memory after errors."""
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def unload(self) -> None:
        """Unload model and free resources."""
        self._voice_prompts.clear()
        if self._model is not None:
            del self._model
            self._model = None
        self._clear_memory()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def capabilities(self) -> Dict[str, object]:
        """Report runtime capabilities for health/status endpoints.

        Consumed by Laravel via GET /health to display engine status.
        """
        return {
            "attn_implementation": self._attn_implementation or "eager",
            "flash_attn_available": FLASH_ATTN_AVAILABLE,
            "streaming_optimizations": self._device == "cuda",
            "device": self._device,
        }
