#!/bin/sh
# Entrypoint for AI Upscale image
#
# Same as base entrypoint, plus:
# - Auto-builds TensorRT engines on first start if missing

set -e

# Fix cache directory ownership
mkdir -p /app/cache
if [ "$(stat -c '%U' /app/cache)" != "netv" ]; then
    chown -R netv:netv /app/cache
fi

# Fix models directory ownership
mkdir -p /models
if [ "$(stat -c '%U' /models)" != "netv" ]; then
    chown -R netv:netv /models
fi

# Add netv user to render device group (for VAAPI hardware encoding)
if [ -e /dev/dri/renderD128 ]; then
    RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
    groupadd --gid "$RENDER_GID" hostrender 2>/dev/null || true
    usermod -aG hostrender netv 2>/dev/null || true
fi

# Build TensorRT engines if missing (first run only)
# Builds both recommended models: 4x-compact (quality) and 2x-liveaction-span (fast)
if ! ls /models/4x-compact_*p_fp16.engine >/dev/null 2>&1; then
    echo "========================================"
    echo "AI Upscale: First start detected"
    echo "========================================"
    echo "Building TensorRT engines for your GPU..."
    echo "Models: 4x-compact (quality), 2x-liveaction-span (fast)"
    echo "This only happens once (cached in /models volume)."
    echo ""
    # Run as netv user so files have correct ownership
    gosu netv env MODEL_DIR=/models MODEL="recommended" /app/tools/install-ai_upscale.sh
fi

# Drop to netv user and run the app
exec gosu netv python3 main.py --port "${NETV_PORT:-8000}" ${NETV_HTTPS:+--https}
