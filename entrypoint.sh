#!/bin/bash

# Install the package in editable mode
# cd /workspace
# uv pip install -e .

# Run the command passed to the container
exec "$@"