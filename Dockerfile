# netv application image
# Uses pre-built FFmpeg base image for fast builds (~2 min vs 15 min)
#
# The FFmpeg base image is built daily with:
# - NVENC (NVIDIA hardware encoding)
# - VAAPI (Intel/AMD hardware encoding)
# - QSV/VPL (Intel QuickSync)
# - All major codecs (x264, x265, VP9, AV1, etc.)
#
# For local builds, set FFMPEG_IMAGE:
#   docker build --build-arg FFMPEG_IMAGE=ghcr.io/jvdillon/netv-ffmpeg:latest .
#
# To use a specific date's snapshot:
#   docker build --build-arg FFMPEG_IMAGE=ghcr.io/jvdillon/netv-ffmpeg:2026-01-04 .

ARG FFMPEG_IMAGE
FROM ${FFMPEG_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install Python (not included in FFmpeg base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# App setup
WORKDIR /app
COPY pyproject.toml README.md ./
COPY *.py ./
COPY templates/ templates/
COPY static/ static/

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir --break-system-packages .

# Runtime config
EXPOSE 8000

ENV NETV_PORT=8000
ENV NETV_HTTPS=""
ENV LOG_LEVEL=INFO

# Create non-root user (entrypoint handles permissions and group membership)
RUN useradd -m netv

# Copy entrypoint and set permissions
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Healthcheck (internal port is always 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/', timeout=5)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
