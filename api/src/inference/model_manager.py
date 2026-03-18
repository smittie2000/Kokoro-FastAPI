"""TTS model management with engine selection."""

from typing import Optional

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        self._backend: Optional[BaseModelBackend] = None
        self._device: Optional[str] = None

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize backend based on configured engine."""
        try:
            self._device = self._determine_device()
            engine = settings.tts_engine
            logger.info(f"Initializing {engine} engine on {self._device}")

            if engine == "qwen3":
                from .qwen3_tts import Qwen3TTS
                self._backend = Qwen3TTS(voices_dir=settings.qwen3_voices_dir)
            else:
                from .kokoro_v1 import KokoroV1
                self._backend = KokoroV1()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize {settings.tts_engine} engine: {e}")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)

        Raises:
            RuntimeError: If initialization fails
        """
        import time

        start = time.perf_counter()

        try:
            # Initialize backend
            await self.initialize()

            # Load model
            engine = settings.tts_engine
            if engine == "qwen3":
                await self.load_model(settings.qwen3_model_path)

                # Warmup with short text using first available voice
                from .qwen3_voice_manager import Qwen3VoiceManager
                voice_mgr = Qwen3VoiceManager(settings.qwen3_voices_dir)
                qwen3_voices = voice_mgr.list_voices()
                if qwen3_voices:
                    warmup_voice = qwen3_voices[0]
                    logger.info(f"Warming up Qwen3-TTS with voice '{warmup_voice}'")
                    async for _ in self.generate("Warmup text.", warmup_voice):
                        pass
                else:
                    logger.warning("No Qwen3 voice profiles found — skipping warmup")

                ms = int((time.perf_counter() - start) * 1000)
                logger.info(f"Warmup completed in {ms}ms")
                return self._device, "qwen3", len(qwen3_voices)
            else:
                model_path = self._config.pytorch_kokoro_v1_file
                await self.load_model(model_path)

                # Use paths module to get voice path
                try:
                    voices = await paths.list_voices()
                    voice_path = await paths.get_voice_path(settings.default_voice)

                    # Warm up with short text
                    warmup_text = "Warmup text for initialization."
                    voice_name = settings.default_voice
                    logger.debug(f"Using default voice '{voice_name}' for warmup")
                    async for _ in self.generate(warmup_text, (voice_name, voice_path)):
                        pass
                except Exception as e:
                    raise RuntimeError(f"Failed to get default voice: {e}")

                ms = int((time.perf_counter() - start) * 1000)
                logger.info(f"Warmup completed in {ms}ms")
                return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("""
Model files not found! You need to download the Kokoro V1 model:

1. Download model using the script:
   python docker/scripts/download_model.py --output api/src/models/v1_0

2. Or set environment variable in docker-compose:
   DOWNLOAD_MODEL=true
""")
            exit(0)
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    def get_backend(self) -> BaseModelBackend:
        """Get initialized backend.

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            async for chunk in self._backend.generate(*args, **kwargs):
                if settings.default_volume_multiplier != 1.0:
                    chunk.audio *= settings.default_volume_multiplier
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return settings.tts_engine


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
