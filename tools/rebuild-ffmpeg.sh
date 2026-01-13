#!/bin/bash
# Rebuild FFmpeg with cudart support
set -e

cd /home/jvdillon/ffmpeg_sources/ffmpeg-snapshot

export PATH="/usr/bin:/bin:/usr/local/bin:/usr/local/cuda/bin:/home/jvdillon/.local/bin:$PATH"
export PKG_CONFIG_PATH="/home/jvdillon/ffmpeg_build/lib/pkgconfig:$PKG_CONFIG_PATH"

./configure \
  --prefix=/home/jvdillon/ffmpeg_build \
  --pkg-config-flags=--static \
  --extra-cflags="-I/home/jvdillon/ffmpeg_build/include -O3 -I/usr/local/cuda/include -I/home/jvdillon/ffmpeg_sources/vulkan-sdk-1.4.335.0/x86_64/include" \
  --extra-cxxflags="-I/home/jvdillon/ffmpeg_sources/libtorch/include -I/home/jvdillon/ffmpeg_sources/libtorch/include/torch/csrc/api/include -I/usr/local/cuda/include" \
  --extra-ldflags="-L/home/jvdillon/ffmpeg_build/lib -s -Wl,-rpath,/home/jvdillon/.local/lib -L/usr/local/cuda/lib64 -L/home/jvdillon/ffmpeg_sources/vulkan-sdk-1.4.335.0/x86_64/lib -L/home/jvdillon/.local/lib -Wl,-rpath,/home/jvdillon/.local/lib" \
  --extra-libs="-lpthread -lm -ldl -lc10_cuda -ltorch_cuda -lcudart" \
  --ld=g++ \
  --bindir=/home/jvdillon/.local/bin \
  --disable-debug \
  --enable-gpl --enable-version3 --enable-openssl --enable-libaom --enable-libass \
  --enable-libbluray --enable-libfdk-aac --enable-libfontconfig --enable-libfreetype \
  --enable-libfribidi --enable-libharfbuzz --enable-libjxl --enable-libmp3lame \
  --enable-libopus --enable-libsvtav1 --enable-libdav1d --enable-libvmaf --enable-libvorbis \
  --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-librubberband \
  --enable-libsoxr --enable-libsrt --enable-libvidstab --enable-libvpl --enable-libzimg \
  --enable-opencl --enable-vaapi --enable-nonfree --enable-cuda-nvcc --enable-nvenc \
  --enable-cuvid --enable-nvdec --enable-amf --enable-libtorch --enable-vulkan \
  --enable-libplacebo --nvccflags='-arch=sm_120'

make -j$(nproc)
make install

echo "BUILD COMPLETE"
