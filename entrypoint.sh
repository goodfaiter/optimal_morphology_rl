#!/bin/bash
# pip3 install /workspace/tools/vlearn/vlearn-0.2.7-cp310-cp310-linux_x86_64.whl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/workspace/.venv/lib/python3.10/site-packages/vlearn/lib"
export CUDA_VISIBLE_DEVICES=0
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Install the package in editable mode
# cd /workspace
# uv pip install -e workspace/

# Run the command passed to the container
exec "$@"