# Upstream Sync Notes

This repo vendors `qwen_tts/` locally instead of installing it via pip. The vendored code combines:

1. **QwenLM/Qwen3-TTS** (upstream) — the base model code
2. **dffdeeq/Qwen3-TTS-streaming** — streaming patches on top

## Periodic Sync Checklist

### QwenLM/Qwen3-TTS (upstream model fixes)
- Repo: https://github.com/QwenLM/Qwen3-TTS
- Check for: decode bug fixes, model architecture updates, new tokenizer versions
- Key files to diff: `qwen_tts/core/`, `qwen_tts/inference/qwen3_tts_tokenizer.py`
- Last synced: 2026-03-19 (includes padding/decode fixes up to commit `6cafe558`)

### dffdeeq/Qwen3-TTS-streaming (streaming patches)
- Repo: https://github.com/dffdeeq/Qwen3-TTS-streaming
- Check for: streaming performance improvements, new optimization methods
- Key files to diff: `qwen_tts/inference/qwen3_tts_model.py`, `qwen_tts/core/models/modeling_qwen3_tts.py`
- Last synced: 2026-03-19 (commit `aad92ef8`)

## Applied Fixes (on top of dffdeeq)

1. **ConvTranspose1d padding** (`core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`)
   - From: QwenLM commit `5f8581d0`
   - Fix: `left_pad=0, right_pad=int(pad)` with guard for empty slice

2. **padding_value=-1 for batch decode** (3 files)
   - From: QwenLM commit `6cafe558`
   - Fix: Use `-1` as padding (not `0` which is a valid codec token), clamp before decode
   - Files: `core/tokenizer_12hz/...`, `core/tokenizer_25hz/...`, `inference/qwen3_tts_tokenizer.py`

3. **transformers 5.x compat** (`core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`)
   - `check_model_inputs` decorator removed in transformers 5.x — replaced with try/except no-op fallback

## How to Sync

```bash
# Clone upstream repos for comparison
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git /tmp/qwen3-upstream
git clone --depth 1 https://github.com/dffdeeq/Qwen3-TTS-streaming.git /tmp/qwen3-streaming

# Diff our vendored code against upstream
diff -r qwen_tts/ /tmp/qwen3-upstream/qwen_tts/ --exclude="__pycache__" | less

# Diff our vendored code against dffdeeq
diff -r qwen_tts/ /tmp/qwen3-streaming/qwen_tts/ --exclude="__pycache__" | less
```

---

## Speaches Comparison (speaches-ai/speaches)

**Repo:** https://github.com/speaches-ai/speaches
**Reviewed:** 2026-03-21
**License:** MIT

Speaches is an OpenAI API-compatible server for both TTS and STT. It aims to be "Ollama for speech."
Evaluated as potential replacement or upstream — conclusion: **keep our fork, use Speaches as reference/inspiration.**

### What Speaches Has That We Don't

| Feature | Speaches | Our Fork |
|---|---|---|
| STT (speech-to-text) | faster-whisper (CTranslate2) + Parakeet (ONNX) | None |
| WebSocket Realtime API | `/v1/realtime` — full OpenAI Realtime spec | None (HTTP streaming only) |
| Dynamic model loading | Per-request, auto-load/unload with configurable TTL | Single engine at startup |
| VAD | Silero VAD v5 (ONNX) | None (LiveKit provides this) |
| Diarization | Pyannote + WeSpeaker speaker embedding | None |
| Observability | Full OpenTelemetry (traces, spans, instrumentation) | None |
| Web UI | Gradio (TTS, STT, audio chat tabs) | None |
| SSE streaming | `stream_format: "sse"` option on `/v1/audio/speech` | Not yet |
| Audio format encoding | ffmpeg subprocess pipe (streaming) | PyAV (in-process) |
| Model registry | HuggingFace-backed, auto-download, filter by task/library/tags | Manual model paths |

### What We Have That Speaches Can't Do

| Feature | Our Fork | Speaches |
|---|---|---|
| Qwen3-TTS | Yes — streaming via dffdeeq patches | No (no ONNX conversion exists) |
| Voice cloning | Zero-shot via ref_audio + ref_text | No (Kokoro pre-baked voices only) |
| PyTorch runtime | torch.compile + CUDA graphs | ONNX-only (no PyTorch at all) |
| vLLM integration path | Possible (PyTorch-based) | Not possible (ONNX-only) |
| MLX (Apple Silicon) | Possible (PyTorch-adjacent) | Not possible (would need CoreML ONNX provider) |
| GPU optimization | torch.compile("reduce-overhead") = 2-3x faster | ONNX CUDAExecutionProvider (no compile optimizations) |

### Key Architectural Difference: ONNX vs PyTorch

Speaches is **ONNX-first**. Every executor runs through ONNX Runtime or CTranslate2:
- Kokoro → `kokoro-onnx` package (ONNX-converted model)
- Piper → ONNX
- Parakeet → `onnx-asr` package
- Whisper → `faster-whisper` (CTranslate2)
- VAD/Diarization → ONNX

Our fork is **PyTorch-first**:
- Qwen3-TTS → PyTorch + transformers + torch.compile
- Kokoro → PyTorch (upstream KPipeline)

This means Speaches **cannot run Qwen3-TTS** without either:
1. Converting Qwen3-TTS to ONNX (nobody has done this for the streaming variant)
2. Adding PyTorch as a dependency (breaks their lightweight philosophy)

Conversely, our fork can't easily adopt their ONNX-based models without adding CTranslate2/onnx-asr.

### Features Worth Borrowing

1. **WebSocket endpoint** — Their `/v1/realtime` implementation uses `asyncio.TaskGroup` + pub/sub event routing.
   Key files: `routers/realtime_ws.py`, `realtime/session.py`, `realtime/pubsub.py`, `realtime/event_router.py`

2. **STT support** — Adding faster-whisper as an STT backend is straightforward. Our fork already has the
   FastAPI plumbing. Key pattern: `SpeechHandler` protocol returns `Generator[Audio]`, same as our streaming.
   Route: `/v1/audio/transcriptions` (OpenAI-compatible)

3. **Dynamic model TTL** — Their `SelfDisposingModel` pattern (ref-counted, auto-unload after TTL) is clean.
   File: `executors/shared/base_model_manager.py` (~110 lines). Could replace our static model loading.

4. **SSE streaming option** — `stream_format: "sse"` sends base64 audio chunks as Server-Sent Events
   instead of raw bytes. Simple addition to our speech router.

5. **Text preprocessing** — They strip emojis and markdown emphasis before TTS (`text_utils.py`).
   We already do abbreviation normalization — could add these too.

### Features We Don't Need (LiveKit Provides)

- **VAD** — LiveKit agents have Silero VAD built-in via `turn_detector`
- **Diarization** — LiveKit tracks participants natively
- **Web UI** — Eloise has Filament admin panel
- **OpenTelemetry** — Nice-to-have but not blocking; Laravel handles tracing

### Deployment Differences

| | Our Fork | Speaches |
|---|---|---|
| Python | 3.14 | Pinned 3.12 |
| Package manager | pip + requirements.txt | uv (with lockfile) |
| Docker base | python:3.14-slim + CUDA | nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 |
| Port | 8880 | 8000 |
| Process manager | uvicorn (direct) | uvicorn (direct) |

### Conclusion

Our fork and Speaches are complementary, not competing:
- **Speaches** = ONNX runtime, broad model support, STT+TTS+WebSocket, community maintained
- **Our fork** = PyTorch runtime, Qwen3-TTS streaming, voice cloning, GPU-optimized, roadmap control

Stripping Speaches' UI, VAD, diarization, and OTel — the core speech serving is similar scope to our fork.
The real gap is **WebSocket support** and **STT**. Both are addable to our fork without architectural changes.

For STT in production: consider running Speaches as a dedicated STT service alongside our TTS fork,
or add faster-whisper directly to our fork (it's CTranslate2, no ONNX Runtime dependency conflict).
