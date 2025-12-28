#!/bin/bash
# Install prerequisites for netv on Debian/Ubuntu
# Run this first on a clean system
set -e

echo "=== Installing system packages ==="
sudo apt update
sudo apt install -y git curl

echo "=== Installing uv ==="
if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Installing Python 3.11 via uv ==="
uv python install 3.11

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Run: ./tools/install-letsencrypt.sh <your-domain>"
echo "  2. Run: ./tools/install-ffmpeg.sh  (optional, for transcoding)"
echo "  3. Run: sudo ./tools/install-netv.sh"
