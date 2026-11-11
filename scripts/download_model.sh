#!/usr/bin/env bash
# scripts/download_model.sh
# Downloads quantized GGUF models from Hugging Face and optionally pushes to Android device.

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "${MODELS_DIR}"

MODEL_NAME="${1:-smollm2}"

case "${MODEL_NAME}" in
    smollm2|smol)
        FILE_NAME="SmolLM2-135M-Instruct-Q4_K_M.gguf"
        URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
        ;;
    llama|llama3.2|llama-1b)
        FILE_NAME="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ;;
    qwen|qwen0.5b)
        FILE_NAME="qwen2.5-0.5b-instruct-q4_k_m.gguf"
        URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ;;
    *)
        echo "Usage: $0 [smollm2 | llama | qwen]"
        exit 1
        ;;
esac

TARGET_PATH="${MODELS_DIR}/${FILE_NAME}"

if [ -f "${TARGET_PATH}" ]; then
    echo "==> Model already exists at: ${TARGET_PATH}"
else
    echo "==> Downloading ${FILE_NAME} (~90MB - ~800MB)..."
    curl -L --progress-bar -o "${TARGET_PATH}" "${URL}"
    echo "==> Download completed: ${TARGET_PATH}"
fi

# Check for connected Android device
if command -v adb >/dev/null 2>&1; then
    DEVICE_COUNT=$(adb devices | grep -v "List of devices" | grep "device$" | wc -l | tr -d ' ')
    if [ "${DEVICE_COUNT}" -gt 0 ]; then
        echo "==> Detected ${DEVICE_COUNT} connected Android device(s)."
        read -r -p "==> Push ${FILE_NAME} to /sdcard/Download/ on device? [Y/n] " response
        response=${response:-Y}
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "==> Pushing to /sdcard/Download/..."
            adb push "${TARGET_PATH}" /sdcard/Download/
            echo "==> Push successful! In NEXA, tap '+' -> 'Import .gguf from storage...' and select it."
        fi
    else
        echo "==> No Android device detected via adb. To push later, run:"
        echo "    adb push ${TARGET_PATH} /sdcard/Download/"
    fi
fi

