# Base python image
FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Install and update system dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y python3-pip git nano mesa-utils

# Install vlearn dependencies
RUN apt update && apt install -y --no-install-recommends 
    curl \
    wget \
    gpg \
    bzip2 \
    ca-certificates \
    git \
    libsdl2-dev \
    libassimp-dev\
    libglu1-mesa-dev \
    g++ \
    libzmq3-dev \ 
    clang\
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Kitware's APT repo (for latest CMake)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | \
    tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | \
    tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake

# Install basic python dependecies
RUN pip3 install --upgrade uv --break-system-packages

# Create python env
# RUN mkdir /workspace
# WORKDIR /workspace
# RUN uv venv --python 3.10 --seed

# Install torch
# RUN pip3 install --break-system-packages torch==2.9.0+cu130 torchvision==0.24.0+cu130 torchaudio==2.9.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# Final dependencies
# RUN pip3 install --break-system-packages matplotlib pyyaml tensorboard h5py

# Add entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]