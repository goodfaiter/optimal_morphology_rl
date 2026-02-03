# optimal_morphology_rl

## To Install
Using docker:

```
cd optimal_morphology_rl
git submodule update --init --recursive
```
*Unfortunately, there is no easy way to add our environment to rl_games_train.py of vlearn. Here we do it manually.*:
```
...
"minimal",
"morph_hand_pen",
"piper_reach_space",
...
elif env == "morph_hand_pen":
    from optimal_morphology_rl.envs.hand_envs.morph_hand_pen_env import MorphHandPenEnvironmentGpu as EnvClass
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '/workspace/tools/vlearn/train/envs/'))
    env_name = "morph_hand_pen-env"
    yml_file = "/workspace/src/optimal_morphology_rl/envs/rl_games_config/morph_hand_pen_ppo.yml"
...
import wandb
wandb.init(project="optimal_morphology_rl", config=config, monitor_gym=True, save_code=True)
runner.run(run_args)
wandb.finish()
...
```

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
