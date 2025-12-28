#!/bin/bash
# Build ffmpeg from source with NVIDIA NVENC support
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
set -e

NPROC=$(nproc)

sudo apt install -y \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  meson \
  nasm \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  libaom-dev \
  libass-dev \
  libdav1d-dev \
  libfdk-aac-dev \
  libffmpeg-nvenc-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  liblzma-dev \
  liblzo2-dev \
  libmp3lame-dev \
  libnuma-dev \
  libopus-dev \
  libsdl2-dev \
  libtool \
  libunistring-dev \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libvpx-dev \
  libx264-dev \
  libx265-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  libxcb1-dev \
  zlib1g-dev

mkdir -p ~/ffmpeg_sources

# libaom
cd ~/ffmpeg_sources && \
git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom && \
mkdir -p aom_build && \
cd aom_build && \
PATH="$HOME/.local/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom && \
PATH="$HOME/.local/bin:$PATH" make -j $NPROC && \
make install

# libsvtav1
cd ~/ffmpeg_sources && \
git -C SVT-AV1 pull 2> /dev/null || git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
mkdir -p SVT-AV1/build && \
cd SVT-AV1/build && \
PATH="$HOME/.local/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
PATH="$HOME/.local/bin:$PATH" make -j $NPROC && \
make install

# libvmaf
cd ~/ffmpeg_sources &&
git -C vmaf-master pull 2> /dev/null || git clone --depth 1 'https://github.com/Netflix/vmaf' 'vmaf-master' &&
mkdir -p 'vmaf-master/libvmaf/build' &&
cd 'vmaf-master/libvmaf/build' &&
meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$HOME/ffmpeg_build" --bindir="$HOME/.local/bin" --libdir="$HOME/ffmpeg_build/lib" &&
ninja &&
ninja install


sudo apt install nv-codec-headers -y ||
(cd ~/ffmpeg_sources &&
git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&
cd nv-codec-headers &&
make &&
make PREFIX=$HOME/ffmpeg_build install)

# Detect CUDA capability
CUDA_FLAGS=""
NVCC_GENCODE=""
if command -v nvidia-smi &> /dev/null; then
  COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  if [ -n "$COMPUTE_CAP" ]; then
    COMPUTE_CAP_NUM=$(echo $COMPUTE_CAP | tr -d '.')
    CUDA_FLAGS="--enable-cuda-nvcc --enable-nvenc --enable-cuvid"
    NVCC_GENCODE="-gencode arch=compute_${COMPUTE_CAP_NUM},code=sm_${COMPUTE_CAP_NUM}"
    echo "Detected NVIDIA GPU with compute capability ${COMPUTE_CAP} (sm_${COMPUTE_CAP_NUM})"
  fi
fi

# ffmpeg
cd ~/ffmpeg_sources
if [ ! -d "ffmpeg" ]; then
  wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
  tar xjvf ffmpeg-snapshot.tar.bz2
fi
cd ffmpeg && \
# Build configure flags
EXTRA_CFLAGS="-I$HOME/ffmpeg_build/include -O3 -march=native -mtune=native"
EXTRA_LDFLAGS="-L$HOME/ffmpeg_build/lib -s"
if [ -n "$CUDA_FLAGS" ]; then
  EXTRA_CFLAGS="$EXTRA_CFLAGS -I/usr/local/cuda/include"
  EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L/usr/local/cuda/lib64"
fi

CONFIGURE_CMD=(
  ./configure
  --prefix="$HOME/ffmpeg_build"
  --pkg-config-flags="--static"
  --extra-cflags="$EXTRA_CFLAGS"
  --extra-ldflags="$EXTRA_LDFLAGS"
  --extra-libs="-lpthread -lm"
  --ld="g++"
  --bindir="$HOME/.local/bin"
  --enable-gpl
  --enable-version3
  --enable-gnutls
  --enable-libaom
  --enable-libass
  --enable-libfdk-aac
  --enable-libfreetype
  --enable-libmp3lame
  --enable-libopus
  --enable-libsvtav1
  --enable-libdav1d
  --enable-libvmaf
  --enable-libvorbis
  --enable-libvpx
  --enable-libx264
  --enable-libx265
  --enable-vaapi
  --enable-nonfree
  $CUDA_FLAGS
)

if [ -n "$NVCC_GENCODE" ]; then
  CONFIGURE_CMD+=(--nvccflags="$NVCC_GENCODE")
fi

PATH="$HOME/.local/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" "${CONFIGURE_CMD[@]}" && \
PATH="$HOME/.local/bin:$PATH" make -j $NPROC && \
make install && \
hash -r

echo "MANPATH_MAP $HOME/.local/bin $HOME/ffmpeg_build/share/man" >> ~/.manpath

# rm -rf ~/ffmpeg_build ~/.local/bin/{ffmpeg,ffprobe,ffplay,x264,x265}
# sed -i '/ffmpeg_build/d' ~/.manpath
# hash -r
# --extra-cflags="-D_GNU_SOURCE"
# cat ~/ffmpeg_sources/ffmpeg/ffbuild/config.log
