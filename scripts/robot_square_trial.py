"""Deploy the latest adapt_finger policy to the robot, run a square fingertip goal and plot the result.

Steps:
  1. Deploy: the newest run's deployment policy ``runs/adapt_finger/<run>/<run>.pt`` (written at the end of training
     from the run's best checkpoint), or ``--policy``, is copied into the hand controller's
     model folder on the robot and the ``ros_dynamixel_hand_controller`` service is rebuilt and started (only
     that service: the high-level controller would publish rl_enabled=False).
  2. Trial: a ROS node in the bag-recorder container publishes the goal (around the ``goal_pose`` rectangle of
     adapt_finger/env.yaml, in its x / z plane) and ``/hand/rl_enabled``, while ``ros2 bag record`` records the
     hand topics. 2 s with RL off at the start corner (the policy history fills), the square, 1 s RL off.
  3. Tip estimate: the robot has no fingertip sensor, so ``tendon_replay.py`` spawns only the hand in vlearn
     (fixed-length tendons, no gravity, no spring / servo / friction) and sets its tendon sliders to the measured
     MCP / PIP spool angles; the finger pose those tendon lengths imply gives the fingertip in the goal frame.
  4. Plots: MCP / PIP spool angle with the desired position (the policy action), spool velocity, and the goal
     and the tip in the goal plane and over time. Files go to ``runs/robot_trials/<policy .pt name>_<name>/`` (``_1``,
     ``_2``, ... appended when it exists), with a short
     ``notes.txt`` (``--notes``: what changed in this experiment; the policy, trial and tip error are added).

Run from the optimal_morphology_rl folder (mcap is not a project dependency):
  uv run --with mcap --with mcap-ros2-support python3 scripts/robot_square_trial.py --name no_fric \
      --notes "servo without M5 friction" [--speed 0.02] [--loops 2] [--pattern points --hold 3]
``--pattern points`` holds the goal at the box middle, top left, top right, bottom right and bottom left instead of the
square; both patterns start at the (x max, z min) corner, the one nearest the open hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO.parent / "runs"
ENV_YAML = REPO / "src" / "optimal_morphology_rl" / "envs" / "adapt_finger" / "env.yaml"

ROBOT = "create@10.241.52.151"
ROBOT_PROJECT = "Documents/projects/actuator"  # relative to the robot user's home
ROBOT_PACKAGE = f"{ROBOT_PROJECT}/ros_dynamixel_hand_controller/ros_dynamixel_hand_controller"  # copied to /colcon_ws/src/<package>
ROBOT_CONTROLLER_SOURCE = f"{ROBOT_PACKAGE}/ros_dynamixel_hand_controller/ros_dynamixel_hand_controller.py"
ROBOT_MODEL = "models/adapt_finger_deployed.pt"  # in the package; the hand controller's default rl_model_path
ROBOT_BAGS = "/home/create/bags"  # mounted at /bags in the bag-recorder container
SERVICE = "ros_dynamixel_hand_controller"
CONTROLLER = "actuator-ros_dynamixel_hand_controller-1"
RECORDER = "actuator-ros-bag-recorder-1"
MCP, PIP = 6, 7  # middle_mcp, middle_pip_dip in the hand controller's joint arrays (its RL_JOINTS)
TOPICS = [
    "/hand/measured_position_rad", "/hand/measured_velocity_rad_per_sec", "/hand/measured_current_amp",
    "/hand/desired_current_amp", "/hand/rl_action_rad", "/hand/rl_enabled", "/pinky_goal_position", "/rosout",
]
PRE_ROLL_SEC, POST_SEC = 2.0, 1.0
APPROACH_SEC = 1.0  # RL starts at the open hand; the swing to the first goal is not tracking error
TIP_SAMPLE_SEC = 0.005  # s, the fingertip is computed from the spool angles at this period
COLORS = {"robot": "#2a78d6", "command": "#eb6834", "goal": "#1baf7a", "tip": "#2a78d6"}

# Goal publisher run on the robot (inside the bag-recorder container, which has rclpy).
RUNNER = textwrap.dedent('''
    import json, sys, time
    import rclpy
    from geometry_msgs.msg import Point
    from rclpy.node import Node
    from std_msgs.msg import Bool


    def park(node, indices, num_joints=12, timeout=8.0):
        """Lower the given joints to just below the tared zero, so the next controller start tares a correct zero.

        The P-hold makes no torque within ~0.5 rad of the target (current deadband), so a target of 0 leaves the spool
        where friction holds it: step each target down (-0.05 rad, then 0.1 rad per 0.5 s, at most -1 rad) until the
        spool reads <= 0, then hold -0.05 rad briefly.
        """
        from std_msgs.msg import Float32MultiArray
        desired_pub = node.create_publisher(Float32MultiArray, "/hand/desired_position_rad", 10)
        state = {}
        node.create_subscription(Float32MultiArray, "/hand/measured_position_rad", lambda m: state.__setitem__("pos", list(m.data)), 10)
        target = {i: -0.05 for i in indices}
        start = last_step = time.monotonic()
        while time.monotonic() - start < timeout:
            rclpy.spin_once(node, timeout_sec=0.0)
            pos = state.get("pos")
            if pos is not None:
                if all(pos[i] <= 0.0 for i in indices):
                    break
                if time.monotonic() - last_step > 0.5:
                    for i in indices:
                        if pos[i] > 0.0:
                            target[i] = max(target[i] - 0.1, -1.0)
                    last_step = time.monotonic()
            desired = [0.0] * num_joints
            for i, value in target.items():
                desired[i] = value
            desired_pub.publish(Float32MultiArray(data=desired))
            time.sleep(0.02)
        end = time.monotonic()
        while time.monotonic() - end < 0.5:  # settle at just below zero
            desired = [0.0] * num_joints
            for i in indices:
                desired[i] = -0.05
            desired_pub.publish(Float32MultiArray(data=desired))
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.02)
        pos = state.get("pos")
        print("parked at " + (", ".join(f"{pos[i]:.3f}" for i in indices) if pos else "unknown") + " rad", flush=True)

    # {"rate": Hz, "pre": s, "post": s, "y": m, "xz": [[x, z], ...] at "rate", "park": [joint indices]}
    path = json.load(open(sys.argv[1]))
    rclpy.init()
    node = Node("square_goal_runner")
    goal_pub = node.create_publisher(Point, "/pinky_goal_position", 10)
    enabled_pub = node.create_publisher(Bool, "/hand/rl_enabled", 10)
    rate, xz = path["rate"], path["xz"]
    pre, post = int(path["pre"] * rate), int(path["post"] * rate)
    start = time.monotonic()
    for k in range(pre + len(xz) + post):
        x, z = xz[min(max(k - pre, 0), len(xz) - 1)]
        goal_pub.publish(Point(x=float(x), y=float(path["y"]), z=float(z)))
        enabled_pub.publish(Bool(data=pre <= k < pre + len(xz)))
        delay = start + (k + 1) / rate - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    for _ in range(20):
        enabled_pub.publish(Bool(data=False))
        time.sleep(0.005)
    print("square done", flush=True)
    park(node, path["park"])
''')


class CommandError(RuntimeError):
    pass


def run(cmd, **kwargs):
    result = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if result.returncode != 0:
        raise CommandError(f"`{' '.join(cmd[:3])} ...` failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


class Robot:
    def __init__(self, host, ssh_options):
        self.host, self.options = host, ssh_options

    def _run(self, tool, args, **kwargs):
        """ssh / scp; if ssh rejects the user's ~/.ssh/config (owner / permissions), retry once without it."""
        try:
            return run([tool, *self.options, *args], **kwargs)
        except CommandError as error:
            if "Bad owner or permissions" not in str(error) or "-F" in self.options:
                raise
            print("WARNING: ssh rejects ~/.ssh/config (bad owner or permissions); ignoring it (-F /dev/null). "
                  "Fix with: chown $(id -un) ~/.ssh/config && chmod 600 ~/.ssh/config")
            self.options = [*self.options, "-F", "/dev/null"]
            return run([tool, *self.options, *args], **kwargs)

    def ssh(self, command, **kwargs):
        return self._run("ssh", ["-o", "BatchMode=yes", self.host, command], **kwargs)

    def check(self):
        """Preflight: reach the robot over ssh (this also applies the ~/.ssh/config fallback before any scp)."""
        try:
            self.ssh("true", timeout=20)
        except (CommandError, subprocess.TimeoutExpired) as error:
            raise SystemExit(f"Cannot reach {self.host} over ssh: {error}")

    # No scp -q: it also silences ssh's own errors (e.g. the ~/.ssh/config one), leaving only "Connection closed".
    def put(self, local, remote):
        self._run("scp", [str(local), f"{self.host}:{remote}"])

    def get(self, remote, local):
        self._run("scp", [f"{self.host}:{remote}", str(local)])

    def ros(self, command):
        """Run a shell command with ROS sourced in the bag-recorder container."""
        inner = f"source /opt/ros/humble/setup.bash; {command}".replace('"', '\\"')
        return self.ssh(f'docker exec {RECORDER} su - ros --whitelist-environment=ROS_DOMAIN_ID -c "{inner}"')


# ---------------------------------------------------------------- deploy


def latest_policy():
    """The newest run's deployment policy runs/adapt_finger/<run>/<run>.pt (its best checkpoint, exported at the end of training)."""
    candidates = [p for p in RUNS.glob("adapt_finger/*/*.pt") if p.stem == p.parent.name]
    for candidate in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if policy_metadata(candidate).get("action_dim") is not None:
            return candidate
    raise SystemExit(f"No runs/adapt_finger/<run>/<run>.pt found under {RUNS}; train a policy first.")


def trial_dir(policy, name):
    """runs/robot_trials/<policy .pt name>_<name>, or with _1, _2, ... appended when that folder already exists."""
    base = f"{Path(policy).stem}_{'_'.join(name.split())}"
    out_dir, k = RUNS / "robot_trials" / base, 0
    while out_dir.exists():
        k += 1
        out_dir = RUNS / "robot_trials" / f"{base}_{k}"
    return out_dir


def run_name(policy):
    """The policy's run, <task>/<timestamp> (e.g. adapt_finger/2026_10_20_14_05_31), for any <run>/*.pt."""
    path = Path(policy).resolve()
    if path.is_relative_to(RUNS.resolve()):
        task, run = path.relative_to(RUNS.resolve()).parts[:2]
        return f"{task}/{run}"
    return path.parent.name


def policy_metadata(path):
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("extra/metadata.json")]
            return json.loads(z.read(names[0])) if names else {}
    except zipfile.BadZipFile:
        return {}


def check_controller_model(robot):
    """The hand controller must load ROBOT_MODEL (its default rl_model_path), or the deployed policy is not run."""
    import re

    source = robot.ssh(f"cat {ROBOT_CONTROLLER_SOURCE}")
    match = re.search(r'p\(\s*"rl_model_path",\s*"/colcon_ws/src/ros_dynamixel_hand_controller/([^"]+)"\s*\)', source)
    loaded = match.group(1) if match else None
    if loaded != ROBOT_MODEL:
        raise SystemExit(
            f"The hand controller loads {loaded!r}, not {ROBOT_MODEL!r}: set its default rl_model_path in "
            f"{ROBOT_CONTROLLER_SOURCE} on the robot to /colcon_ws/src/ros_dynamixel_hand_controller/{ROBOT_MODEL}."
        )


def deploy(robot, policy, timeout=180.0):
    check_controller_model(robot)
    target = f"{ROBOT_PACKAGE}/{ROBOT_MODEL}"
    print(f"Deploying {policy} -> {target} ...")
    robot.put(policy, target)
    robot.ssh(f"cd ~/{ROBOT_PROJECT} && HOST_UID=$(id -u) docker compose up -d --build {SERVICE}", timeout=900)
    started = time.monotonic()
    while time.monotonic() - started < timeout:  # the controller winds the tendons and tares the zero at startup
        if "zero tared" in robot.ssh(f"docker logs --since 5m {CONTROLLER} 2>&1 | tail -n 50"):
            return
        time.sleep(2.0)
    raise SystemExit(f"{CONTROLLER} did not report 'zero tared' within {timeout:.0f} s; check `docker logs {CONTROLLER}`.")


# ---------------------------------------------------------------- trial


def pattern_label(args):
    if args.pattern == "points":
        return f"points ({', '.join(POINTS)}), {args.hold:g} s each"
    return f"square at {args.speed * 100:.1f} cm/s, {args.loops} loop(s)"


def square_path(rate, speed, loops):
    """Goal x / z around the goal_pose rectangle (counterclockwise from the (x max, z min) corner), at `speed` m/s."""
    from optimal_morphology_rl.utils.config import load_yaml_with_context

    goal = load_yaml_with_context(ENV_YAML, context={"mode": "play"})["goal_pose"]["goal_position"]
    (x0, x1), (z0, z1), y = sorted(goal["x"]), sorted(goal["z"]), float(goal["y"][0])
    corners = np.array([[x1, z0], [x1, z1], [x0, z1], [x0, z0], [x1, z0]])
    edges = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    s = np.arange(0.0, loops * cum[-1], speed / rate) % cum[-1]
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, 3)
    frac = ((s - cum[i]) / edges[i])[:, None]
    return corners[i] + frac * (corners[i + 1] - corners[i]), y


POINTS = ("middle", "top left", "top right", "bottom right", "bottom left")
START_HOLD_SEC = 2.0  # points pattern: hold at the start corner before the first point


def points_path(rate, hold):
    """Goal x / z held at the box start corner (x max, z min), then at each of POINTS for `hold` s."""
    from optimal_morphology_rl.utils.config import load_yaml_with_context

    goal = load_yaml_with_context(ENV_YAML, context={"mode": "play"})["goal_pose"]["goal_position"]
    (x0, x1), (z0, z1), y = sorted(goal["x"]), sorted(goal["z"]), float(goal["y"][0])
    xy = {"middle": ((x0 + x1) / 2, (z0 + z1) / 2), "top left": (x0, z1), "top right": (x1, z1),
          "bottom right": (x1, z0), "bottom left": (x0, z0)}
    targets = [(x1, z0, START_HOLD_SEC)] + [(*xy[name], hold) for name in POINTS]
    return np.concatenate([np.tile([x, z], (int(duration * rate), 1)) for x, z, duration in targets]), y


def run_trial(robot, out_dir, args):
    rate = 200.0
    if args.pattern == "points":
        xz, y = points_path(rate, args.hold)
    else:
        xz, y = square_path(rate, args.speed, args.loops)
    name = out_dir.name
    remote_dir = f"{ROBOT_BAGS}/robot_trials"
    robot.ssh(f"mkdir -p {remote_dir}")
    (out_dir / "goal_path.json").write_text(json.dumps(
        {"rate": rate, "pre": PRE_ROLL_SEC, "post": POST_SEC, "y": y, "xz": xz.tolist(), "park": [MCP, PIP]}))
    (out_dir / "run_square.py").write_text(RUNNER)
    robot.put(out_dir / "goal_path.json", f"{remote_dir}/{name}_goal_path.json")
    robot.put(out_dir / "run_square.py", f"{remote_dir}/run_square.py")

    bag = f"/bags/robot_trials/{name}"
    print(f"Running the {pattern_label(args)}: {len(xz) / rate:.1f} s ...")
    robot.ros(f"rm -rf {bag}; nohup ros2 bag record -s mcap -o {bag} {' '.join(TOPICS)} > {bag}.log 2>&1 &")
    time.sleep(2.0)
    try:
        output = robot.ros(f"python3 /bags/robot_trials/run_square.py /bags/robot_trials/{name}_goal_path.json")
        for line in output.splitlines():
            if line.startswith("parked"):
                print(f"Finger {line}")
    finally:
        time.sleep(1.0)
        robot.ros(f"pkill -INT -f 'ros2 bag record -s mcap -o {bag}' || true")
        time.sleep(3.0)
    mcap_name = robot.ssh(f"ls {ROBOT_BAGS}/robot_trials/{name}/").split()
    mcap_name = [n for n in mcap_name if n.endswith(".mcap")][0]
    local = out_dir / mcap_name
    robot.get(f"{ROBOT_BAGS}/robot_trials/{name}/{mcap_name}", local)
    return local


def read_bag(path):
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        raise SystemExit("Needs mcap: run with  uv run --with mcap --with mcap-ros2-support python3 scripts/robot_square_trial.py")
    data, logs = {}, []
    with open(path, "rb") as f:
        for schema, channel, message, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            t = message.publish_time * 1e-9
            if channel.topic == "/rosout":
                logs.append((msg.level, msg.msg))
                continue
            if schema.name == "std_msgs/msg/Float32MultiArray":
                value = list(msg.data)
            elif schema.name == "geometry_msgs/msg/Point":
                value = [msg.x, msg.y, msg.z]
            elif schema.name == "std_msgs/msg/Bool":
                value = [float(msg.data)]
            else:
                continue
            data.setdefault(channel.topic, ([], []))
            data[channel.topic][0].append(t)
            data[channel.topic][1].append(value)
    arrays = {topic: (np.array(t), np.array(v, dtype=float)) for topic, (t, v) in data.items()}
    for level, text in logs:
        if level >= 30:  # warnings and errors of the run
            print(f"  robot log: {text}")
    return arrays


def trial_signals(bag):
    """Signals of the RL segment, time since RL on; ``sig.desired`` is the desired spool angle (MCP, PIP)."""
    t_en, en = bag["/hand/rl_enabled"]
    on = t_en[en[:, 0] > 0.5]
    t0, t1 = on[0], on[-1]
    t_pos, pos = bag["/hand/measured_position_rad"]
    t_vel, vel = bag["/hand/measured_velocity_rad_per_sec"]
    t_goal, goal = bag["/pinky_goal_position"]
    m = (t_pos >= t0) & (t_pos <= t1)
    t = t_pos[m] - t0
    sig = SimpleNamespace(
        t=t,
        theta=pos[m][:, [MCP, PIP]],
        omega=np.stack([np.interp(t_pos[m], t_vel, vel[:, j]) for j in (MCP, PIP)], -1),
        goal=np.stack([np.interp(t_pos[m], t_goal, goal[:, j]) for j in range(3)], -1),
    )
    if "/hand/rl_action_rad" in bag:
        t_act, act = bag["/hand/rl_action_rad"]
        sig.t_action, sig.desired = t_act - t0, act[:, :2]  # the policy action is the desired spool angle
    rate = 1.0 / np.median(np.diff(t_pos))
    print(f"RL segment {t[-1]:.1f} s, state at {rate:.0f} Hz")
    return sig


# ---------------------------------------------------------------- tip estimate


def estimate_tip(sig, out_dir):
    """Fingertip (goal frame) for the measured spool angles: only the hand, its fixed-length tendons moved (tendon_replay)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tendon_replay import TendonReplay, fixed_tendon_vsim, task_vsim

    print("Estimating the fingertip: moving the tendons of the sim hand to the measured spool angles ...")
    replay = TendonReplay(fixed_tendon_vsim(task_vsim(), out_dir / "replay_hand.vsim"))
    t = np.arange(0.0, sig.t[-1], TIP_SAMPLE_SEC)
    theta = np.stack([np.interp(t, sig.t, sig.theta[:, j]) for j in range(2)], -1)
    return SimpleNamespace(t=t, tip=replay.tip(theta))


# ---------------------------------------------------------------- plots


def plot(sig, replay, out_dir, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 6.4), sharex=True)
    for j, joint in enumerate(["MCP", "PIP"]):
        ax = axes[0, j]
        if hasattr(sig, "desired"):
            # The servo current is motor_position_gain * (desired - position).
            ax.plot(sig.t_action, sig.desired[:, j], color=COLORS["command"], lw=0.8, label="desired (policy action)")
        ax.plot(sig.t, sig.theta[:, j], color=COLORS["robot"], lw=1.4, label="measured")
        ax.set_title(f"{joint} spool angle", loc="left", fontsize=10)
        ax.set_ylabel("rad")
        ax.legend(loc="upper right", fontsize=8)
        ax = axes[1, j]
        ax.plot(sig.t, sig.omega[:, j], color=COLORS["robot"], lw=1.0)
        ax.set_title(f"{joint} spool velocity (measured)", loc="left", fontsize=10)
        ax.set_ylabel("rad/s")
        ax.set_xlabel("time since RL on [s]")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    joints_png = out_dir / "spools.png"
    fig.savefig(joints_png, dpi=90)

    tip = replay.tip
    goal = np.stack([np.interp(replay.t, sig.t, sig.goal[:, j]) for j in range(3)], -1)
    err = np.linalg.norm(tip[:, [0, 2]] - goal[:, [0, 2]], axis=1) * 1000
    # The first APPROACH_SEC: RL starts at the open hand (spools at the tared 0, straight finger, tip near
    # x = 86, z = -3 mm) and the finger swings to the first goal; left out of the tip plots and the tip error.
    approach = replay.t < APPROACH_SEC
    tracking = sig.t >= APPROACH_SEC
    fig = plt.figure(figsize=(15, 6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4])
    ax = fig.add_subplot(grid[:, 0])
    ax.plot(sig.goal[tracking, 0] * 1000, sig.goal[tracking, 2] * 1000, color=COLORS["goal"], lw=2.0, label="goal")
    ax.plot(tip[~approach, 0] * 1000, tip[~approach, 2] * 1000, color=COLORS["tip"], lw=1.2,
            label=f"tip (from the measured spool angles), after the {APPROACH_SEC:.0f} s approach")
    ax.set_xlabel("x in Pinky_Abd [mm]")
    ax.set_ylabel("z in Pinky_Abd [mm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{title}\nmean tip error after {APPROACH_SEC:.0f} s: {err[~approach].mean():.1f} mm "
                 f"(median {np.median(err[~approach]):.1f} mm)", loc="left", fontsize=9)
    for row, (axis, name) in enumerate(((0, "x"), (2, "z"))):
        ax = fig.add_subplot(grid[row, 1])
        ax.plot(sig.t[tracking], sig.goal[tracking, axis] * 1000, color=COLORS["goal"], lw=2.0, label="goal (desired)")
        ax.plot(replay.t[~approach], tip[~approach, axis] * 1000, color=COLORS["tip"], lw=1.2, label="tip")
        ax.set_ylabel(f"{name} [mm]")
        ax.set_title(f"tip {name} in Pinky_Abd", loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_xlabel("time since RL on [s]")
    fig.tight_layout()
    xy_png = out_dir / "tip_xz.png"
    fig.savefig(xy_png, dpi=90)
    np.savez(out_dir / "signals.npz", t=sig.t, theta=sig.theta, omega=sig.omega, goal=sig.goal,
             t_tip=replay.t, tip=tip, tip_err_mm=err)
    print(f"Mean tip error after the {APPROACH_SEC:.0f} s approach: {err[~approach].mean():.1f} mm (median {np.median(err[~approach]):.1f} mm)")
    return (joints_png, xy_png), (err[~approach].mean(), np.median(err[~approach])), (replay.t, err)


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", type=Path, help="deployment policy (default: the newest runs/adapt_finger/<run>/<run>.pt, its best checkpoint)")
    parser.add_argument("--speed", type=float, default=0.02, help="goal speed along the square [m/s]")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--pattern", choices=("square", "points"), default="square")
    parser.add_argument("--hold", type=float, default=3.0, help="points pattern: hold per point [s]")
    parser.add_argument("--robot", default=ROBOT)
    parser.add_argument("--ssh-option", action="append", default=[], help="extra ssh/scp options, e.g. --ssh-option='-F /dev/null'")
    parser.add_argument("--skip-deploy", action="store_true", help="use the policy already running on the robot")
    parser.add_argument("--keep-running", action="store_true", help="leave the hand controller running afterwards")
    parser.add_argument("--bag", type=Path, help="analyze an existing trial bag (.mcap) instead of running the robot")
    parser.add_argument("--name", help="short descriptor appended to the trial folder, e.g. no_fric (required for a new trial)")
    parser.add_argument("--notes", default="", help="what changed in this experiment, for the folder's notes.txt")
    args = parser.parse_args()

    if not args.bag and not args.name:
        parser.error("--name is required for a new trial (a short descriptor, e.g. no_fric)")
    robot = Robot(args.robot, [option for item in args.ssh_option for option in item.split()])

    policy = args.policy or latest_policy()
    # A re-analysis (--bag) writes its plots next to the bag, a new trial into a new folder named after the policy:
    # <policy .pt name>_<name>, with _1, _2, ... appended when that folder exists.
    if args.bag:
        out_dir = args.bag.parent
    else:
        out_dir = trial_dir(policy, args.name)
    title = f"{run_name(policy)}: {pattern_label(args)}"
    if args.bag:
        bag_path = args.bag
    else:
        meta = policy_metadata(policy)
        print(f"Policy {policy} ({meta.get('network', 'traced_model')}, {meta.get('frequency')} Hz)")
        out_dir.mkdir(parents=True, exist_ok=True)
        robot.check()
        try:
            if not args.skip_deploy:
                deploy(robot, policy)
            bag_path = run_trial(robot, out_dir, args)
        finally:
            if not args.keep_running:
                try:  # best effort: do not hide the error that ended the trial
                    robot.ssh(f"cd ~/{ROBOT_PROJECT} && HOST_UID=$(id -u) docker compose stop {SERVICE}")
                    print("Hand controller stopped.")
                except CommandError as error:
                    print(f"WARNING: could not stop the hand controller: {error}")
    sig = trial_signals(read_bag(bag_path))
    replay = estimate_tip(sig, out_dir)
    pngs, (mean_err, median_err), (t_tip, err) = plot(sig, replay, out_dir, title)
    for png in pngs:
        print(f"Plot: {png}")
    settled = {}
    if args.pattern == "points":
        # Settled error per point: the mean over the last 1 s of its hold (time since RL on).
        for k, name in enumerate(POINTS):
            end = START_HOLD_SEC + (k + 1) * args.hold
            window = (t_tip >= end - 1.0) & (t_tip < end)
            settled[name] = float(err[window].mean()) if window.any() else float("nan")
        print("Settled tip error (last 1 s of each hold): " + ", ".join(f"{n} {e:.1f} mm" for n, e in settled.items()))
    if not args.bag:  # a re-analysis keeps the trial's notes
        lines = [args.notes] if args.notes else []
        lines += [
            f"policy: {policy} (run {run_name(policy)})",
            f"pattern: {pattern_label(args)}",
            f"robot tip error after the {APPROACH_SEC:.0f} s approach: mean {mean_err:.1f} mm, median {median_err:.1f} mm",
        ]
        if settled:
            lines.append("settled tip error per point (last 1 s of each hold): "
                         + ", ".join(f"{n} {e:.1f} mm" for n, e in settled.items()))
        (out_dir / "notes.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
