#!/usr/bin/env bash
# Zen Guard identity fine-tuning via zoo-gym
# Model: zenlm/zen-guard (8B dense, Qwen3, safety classifier)
# Method: LoRA SFT, rank 8
# Hardware: 16GB+ VRAM recommended (bf16), or 8GB with QLoRA

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${REPO_ROOT}/training/config.yaml"

echo "=== Zen Guard Identity Training ==="
echo "Repo:   ${REPO_ROOT}"
echo "Config: ${CONFIG}"

if ! command -v gym &>/dev/null && ! python -m gym.launcher --help &>/dev/null 2>&1; then
    echo "zoo-gym not found. Install with: pip install zoo-gym"
    exit 1
fi

cd "${REPO_ROOT}"

if command -v gym &>/dev/null; then
    gym train "${CONFIG}"
else
    python -m gym.launcher train "${CONFIG}"
fi

echo ""
echo "=== Training complete ==="
echo "Output: ${REPO_ROOT}/training/output"
echo ""
echo "To merge and push:"
echo "  gym export --model_name_or_path zenlm/zen-guard \\"
echo "    --adapter_name_or_path training/output \\"
echo "    --export_dir training/merged \\"
echo "    --export_size 2"
echo "  hf upload zenlm/zen-guard training/merged"
