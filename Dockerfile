# Base python image
FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

RUN sed -i 's/htt[p|ps]:\/\/archive.ubuntu.com\/ubuntu\//mirror:\/\/mirrors.ubuntu.com\/mirrors.txt/g' /etc/apt/sources.list

# Install and update system dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y --no-install-recommends python3-pip git nano mesa-utils

# Install vlearn dependencies
RUN apt update && apt install -y --no-install-recommends \
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
    clang

# Install basic python dependecies
RUN pip3 install --upgrade uv --break-system-packages

# Install PythonSCAD
RUN apt update && apt install -y --no-install-recommends lsb-release
RUN wget -qO - https://repos.pythonscad.org/apt/pythonscad-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/pythonscad-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/pythonscad-archive-keyring.gpg] https://repos.pythonscad.org/apt $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/pythonscad.list
RUN apt update && apt install -y pythonscad

# Create a profile.d script for interactive shells
COPY setenv.sh /docker-entrypoint.d/
RUN chmod +x /docker-entrypoint.d/setenv.sh
RUN echo 'source /docker-entrypoint.d/setenv.sh' >> /etc/profile
RUN echo 'source /docker-entrypoint.d/setenv.sh' >> /root/.bashrc

RUN apt update && apt clean && rm -rf /var/lib/apt/lists/*

# Add entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]