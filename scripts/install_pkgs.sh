#!/bin/bash
# Installs Python dependencies at the start of a Claude Code CLOUD session only.
# On your own laptop (VS Code) this exits immediately and installs nothing.
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR" || exit 0
export PIP_BREAK_SYSTEM_PACKAGES=1

# Skip if already installed in this VM (keeps resumed sessions fast).
if python -c "import torch, pydantic, fastapi, onnxruntime" 2>/dev/null; then
  exit 0
fi

# torch should already be present (CPU build) from the environment's setup script.
# torch>=2.6 in requirements.txt is then satisfied and no CUDA wheels are pulled.
pip install -q -r requirements.txt || echo "WARNING: pip install -r requirements.txt failed - ask Claude to investigate"
exit 0
