#!/usr/bin/env python3
"""Download Kokoro v1.0 model and voices from HuggingFace."""

import os

from huggingface_hub import snapshot_download
from loguru import logger

REPO_ID = "hexgrad/Kokoro-82M"


def download_model(output_dir: str) -> None:
    """Download model files from HuggingFace Hub.

    Downloads kokoro-v1_0.pth, config.json, and voice .pt files
    into the specified output directory.

    Args:
        output_dir: Directory to save model files
    """
    model_path = os.path.join(output_dir, "kokoro-v1_0.pth")

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        logger.info("Model files already exist, skipping download")
        return

    logger.info(f"Downloading {REPO_ID} to {output_dir}")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=output_dir,
        allow_patterns=["kokoro-v1_0.pth", "config.json"],
    )

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found after download: {model_path}")

    logger.info(f"Model files prepared in {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Kokoro v1.0 model")
    parser.add_argument(
        "--output", required=True, help="Output directory for model files"
    )

    args = parser.parse_args()
    download_model(args.output)


if __name__ == "__main__":
    main()
