#!/bin/bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/workspace/.venv/lib/python3.10/site-packages/vlearn/lib"
export CUDA_VISIBLE_DEVICES=0
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export WANDB_API_KEY=
export VL_TURBO_ACTIVATE_PATH=/workspace/tools/vlearn/TurboActivate.dat
export VL_LICENSE_KEY_PATH=/vsim_tech/License.key