#!/bin/bash
# Download Kokoro v1.0 model from HuggingFace Hub
# Wrapper around download_model.py for use in Dockerfiles

set -e

# Find project root by looking for api directory
find_project_root() {
    local current_dir="$PWD"
    local max_steps=5
    local steps=0

    while [ $steps -lt $max_steps ]; do
        if [ -d "$current_dir/api" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
        ((steps++))
    done

    echo "Error: Could not find project root (no api directory found)" >&2
    exit 1
}

PROJECT_ROOT=$(find_project_root)
MODEL_DIR="$PROJECT_ROOT/api/src/models/v1_0"

echo "Downloading Kokoro model to: $MODEL_DIR"
python "$PROJECT_ROOT/download_model.py" --output "$MODEL_DIR"
