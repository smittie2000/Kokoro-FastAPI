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
