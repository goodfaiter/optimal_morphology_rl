# optimal_morphology_rl

## To Install
Using docker:

```
docker compose up dev_gpu -d --build
```


## To Run
### Via Terminal
In your favorite terminal:

```
export __GLX_VENDOR_LIBRARY_NAME=nvidia && docker exec -it optimal_morphology_rl-dev-gpu-1 bash
uv run python3 rl_games_train.py morph_hand_pen train --headless False
uv run python3 rl_games_train.py morph_hand_pen play <path>
```

### Via VSCode
Or a *better* way is to setup your `.vscode/launch.json`:

```
"version": "0.2.0",
"configurations": [
    {
        "name": "Train",
        "type": "debugpy",
        "request": "launch",
        "program": "/workspace/tools/vlearn/train/rl_games_train.py",
        "console": "integratedTerminal",
        "cwd": "/workspace/tools/vlearn/train",
        "env": {
            "CUDA_VISIBLE_DEVICES": "0",
            "__NV_PRIME_RENDER_OFFLOAD": "1",
            "__NV_PRIME_RENDER_OFFLOAD_PROVIDER": "nvidia",
            "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}:/workspace/.venv/lib/python3.10/site-packages/vlearn/lib",
            "WANDB_API_KEY": ""
        },
        "args": [
            "morph_hand_pen",
            "train",
            "--headless", "False"
        ]
    }
]
```


## Notes
- Sometimes the X-host and GPU access gets wonky and the container needs to be restarted.
- Check `glxgears` in vscode terminal to check if everything works.
