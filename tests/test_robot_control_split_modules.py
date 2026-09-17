"""Unit tests for the split robot_control compute/apply modules."""

import pytest
import torch
import vlearn as v

from optimal_morphology_rl.modules.apply_control import (
    ApplyMotorForcesModule,
    ApplyRootForcesModule,
    ApplyRootVelocityModule,
    ApplyTendonForcesModule,
)
from optimal_morphology_rl.modules.gravity_compensation_module import GravityCompensationModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.randomize_friction_module import RandomizeFrictionModule
from optimal_morphology_rl.modules.robot_control import (
    RobotControlFloatingHandModule,
    RobotControlMotorsModule,
    RobotControlTendonsModule,
)


class _FakeEnv:
    pass


class _FakeMotorDef:
    def __init__(self, name: str):
        self.name = name


class _FakeTendonDef:
    def __init__(self, name: str):
        self.name = name


class _FakeArtDef:
    def __init__(self, motor_names, tendon_names=None):
        self._motor_names = motor_names
        self._tendon_names = tendon_names or []

    def get_motor_def(self, i: int) -> _FakeMotorDef:
        return _FakeMotorDef(self._motor_names[i])

    def get_spatial_tendon_def(self, i: int) -> _FakeTendonDef:
        return _FakeTendonDef(self._tendon_names[i])


class _FakeRobot:
    def __init__(self, fixed_hand: bool, use_tendon: bool, num_tendons: int = 2, num_motors: int = 4, tendon_names=None):
        if tendon_names is None:
            tendon_names = [f"tendon_{i}" for i in range(num_tendons)]
        self.art_def = _FakeArtDef(["mcp", "pip", "abd", "lum"], tendon_names=tendon_names)
        self.use_tendon = use_tendon
        self.fixed_hand = fixed_hand
        self.num_tendons = num_tendons if use_tendon else 0
        self.num_motors = num_motors
        self.num_links = 1
        self.motor_to_joint_dof_index = torch.arange(num_motors, dtype=torch.long)
        self.arti_handle = 99
        self.rigid_mat_handle = 12
        self.get_root_transform_buf = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3]])


class _FakeGym:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def create_gpu_array(self, cmds):
        return ("gpu_array", cmds[0])

    def set_spatial_tendon_forces(self, arr):
        self.calls.append(("set_spatial_tendon_forces", arr))

    def set_motor_forces(self, arr):
        self.calls.append(("set_motor_forces", arr))

    def set_articulation_kinematic_states(self, arr):
        self.calls.append(("set_articulation_kinematic_states", arr))

    def set_link_external_forces(self, arr):
        self.calls.append(("set_link_external_forces", arr))

    def set_rigid_material_properties(self, arr):
        self.calls.append(("set_rigid_material_properties", arr))


class _FakeEnvGroup:
    def __init__(self, container):
        self._container = container

    def create_spatial_tendon_control_command(self, buf, handle):
        return ("tendon_cmd", buf)

    def create_motor_control_command(self, buf, handle, index_range=None):
        return ("motor_cmd", buf)

    def create_articulation_kinematic_state_command(self, *args, **kwargs):
        return ("kin_cmd", args)

    def create_link_external_force_command(self, buf, handle, link_range, force_type=None):
        return ("force_cmd", buf)

    def create_rigid_material_property_command(self, prop, buf, handle, mask):
        return ("friction_cmd", buf)


@pytest.fixture
def container(monkeypatch) -> ModuleContainer:
    # wrap_gpu_buffer asserts cuda tensors; tests run on cpu with fake commands.
    monkeypatch.setattr(v, "wrap_gpu_buffer", lambda buf: ("wrapped", buf))
    cont = ModuleContainer()
    cont.total_num_envs = 1
    cont.device = torch.device("cpu")
    cont.env = _FakeEnv()
    cont.create_robot_config = {}
    cont.gym = _FakeGym()
    cont.env_group = _FakeEnvGroup(cont)
    cont.idx_handle = 7
    cont.robot = _FakeRobot(fixed_hand=False, use_tendon=True)
    cont.robot.get_root_transform_buf = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3]])
    cont.reset_buf = torch.zeros(1, dtype=torch.bool)
    cont.inverse_reset_buf = torch.ones(1, dtype=torch.bool)
    cont.act_buf = torch.zeros((1, 8), dtype=torch.float32)
    cont.scaled_act_buf = torch.zeros((1, 8), dtype=torch.float32)
    cont.scaled_act_buf[:, 6:8] = 0.5
    return cont


# ---------------------------------------------------------------------------
# Compute modules
# ---------------------------------------------------------------------------
def test_floating_hand_finalize_validates_hand_type(container: ModuleContainer) -> None:
    module = RobotControlFloatingHandModule({})
    module.finalize(container)

    container.robot = _FakeRobot(fixed_hand=True, use_tendon=True)
    module = RobotControlFloatingHandModule({})
    with pytest.raises(RuntimeError, match="floating"):
        module.finalize(container)


def test_floating_hand_step_writes_world_frame_velocities(container: ModuleContainer) -> None:
    container.robot = _FakeRobot(fixed_hand=False, use_tendon=True)

    RobotControlFloatingHandModule({}).finalize(container)

    container.scaled_act_buf[:, :] = 0.0
    container.scaled_act_buf[:, :2] = 1.0  # local root linear velocities

    # set_root buffers are owned by apply_root_velocity (allocated for the test).
    container.set_root_transform_buf = torch.zeros((1, 7))
    container.set_root_vel_buf = torch.zeros((1, 6))

    module = RobotControlFloatingHandModule({})
    module.step(container)

    assert torch.allclose(container.set_root_transform_buf, container.robot.get_root_transform_buf)
    # Identity quaternion: local velocities pass through.
    assert torch.allclose(container.set_root_vel_buf[:, :2], torch.ones((1, 2)))
    assert torch.allclose(container.set_root_vel_buf[:, 2:], torch.zeros((1, 4)))


def test_tendons_module_composes_forces(container: ModuleContainer) -> None:
    module = RobotControlTendonsModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.set_tendon_controls_buf = torch.full((1, 2), -5.0)
    container.set_motor_cmd_buf = torch.full((1, 4), 3.0)

    module.step(container)

    # Policy actions clamped to >= 0 on the tilt/tendon slice.
    assert torch.allclose(container.set_tendon_controls_buf, torch.full((1, 2), 0.5))
    # The tendon module no longer touches the motor command buffer.
    assert torch.allclose(container.set_motor_cmd_buf, torch.full((1, 4), 3.0))


def test_tendons_module_applies_model_and_rigid_overrides(container: ModuleContainer) -> None:
    module = RobotControlTendonsModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.set_tendon_controls_buf = torch.full((1, 2), 0.5)
    container.set_force_torque_buf = torch.zeros((1, 1, 6))

    container.tendon_force_buf = torch.tensor([[10.0, 20.0]])
    container.tendon_model_indices = torch.tensor([0, 1])
    container.rigid_tendon_force_buf = torch.tensor([[1.0, 2.0]])
    container.rigid_tendon_indices = torch.tensor([0, 1])

    module.step(container)

    # Model forces override the clamped actions; rigid forces accumulate on top.
    assert torch.allclose(container.set_tendon_controls_buf, torch.tensor([[11.0, 22.0]]))


def test_tendons_module_requires_tendon_robot(container: ModuleContainer) -> None:
    container.robot = _FakeRobot(fixed_hand=False, use_tendon=False)
    module = RobotControlTendonsModule({})
    with pytest.raises(RuntimeError, match="tendon-driven"):
        module.finalize(container)


def test_mask_excludes_fixed_tendons_from_policy_actions(container: ModuleContainer) -> None:
    container.robot = _FakeRobot(
        fixed_hand=False, use_tendon=True, num_tendons=3,
        tendon_names=["mcp_tendon", "pip_tendon", "dip_tendon"],
    )
    container.create_robot_config = {"fixed_tendon_substrings": ["dip"]}
    container.num_actions = None  # reset the guard so the action space is (re)built

    module = RobotControlTendonsModule({})
    module.finalize(container)

    assert container.num_active_dofs == 2
    assert torch.equal(container.active_dof_mask, torch.tensor([True, True, False]))
    assert torch.equal(container.active_dof_indices.to(torch.long), torch.tensor([0, 1]))
    assert container.active_dof_slice == slice(6, 8)
    assert container.num_actions == 8  # 6 root + 2 policy-controlled tendons


def test_mask_includes_all_tendons_without_config(container: ModuleContainer) -> None:
    module = RobotControlTendonsModule({})
    module.finalize(container)

    assert container.num_active_dofs == 2
    assert torch.equal(container.active_dof_indices.to(torch.long), torch.tensor([0, 1]))


def test_tendons_module_scatter_leaves_fixed_tendon_zero(container: ModuleContainer) -> None:
    container.robot = _FakeRobot(
        fixed_hand=False, use_tendon=True, num_tendons=3,
        tendon_names=["mcp_tendon", "pip_tendon", "dip_tendon"],
    )
    container.create_robot_config = {"fixed_tendon_substrings": ["dip"]}
    container.num_actions = None

    module = RobotControlTendonsModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.scaled_act_buf = torch.zeros((1, 8))
    container.scaled_act_buf[:, 6:8] = 0.5

    container.set_tendon_controls_buf = torch.full((1, 3), 9.0)
    container.rigid_tendon_force_buf = torch.tensor([[0.0, 0.0, 3.0]])
    container.rigid_tendon_indices = torch.tensor([-1])

    module.step(container)
    module.step(container)

    # Policy columns (MCP/PIP) get the clamped actions...
    assert torch.allclose(container.set_tendon_controls_buf[:, :2], torch.full((2,), 0.5))
    # ...while the fixed DIP tendon only receives the rigid force, with no
    # cross-step accumulation despite the repeated rigid add.
    assert container.set_tendon_controls_buf[:, -1].item() == pytest.approx(3.0)


def test_motors_module_writes_policy_force_buffer(container: ModuleContainer) -> None:
    container.robot = _FakeRobot(fixed_hand=False, use_tendon=False)
    module = RobotControlMotorsModule({})
    module.finalize(container)
    module.post_finalize(container)

    # Motor hand: 4 motors (mcp, pip, abd, lum); abd is passive -> 3 active.
    # Root slice (6) offsets the active motor actions.
    container.scaled_act_buf = torch.zeros((1, 10))
    container.scaled_act_buf[:, :6] = -0.5
    container.scaled_act_buf[:, 6:9] = 2.0

    assert container.motor_policy_force_buf.shape == (1, 4)
    assert torch.all(container.motor_policy_force_buf == 0.0)

    module.step(container)

    active = container.active_dof_mask.nonzero().flatten()  # [0, 1, 3]
    inactive = torch.ones(4, dtype=torch.bool)
    inactive[active] = False

    assert torch.allclose(container.motor_policy_force_buf[0, active], torch.full(active.shape, 2.0))
    assert torch.all(container.motor_policy_force_buf[0, inactive] == 0.0)


def test_motors_module_requires_motor_robot(container: ModuleContainer) -> None:
    module = RobotControlMotorsModule({})
    with pytest.raises(RuntimeError, match="motor-driven"):
        module.finalize(container)


def test_compute_modules_build_action_space_once(container: ModuleContainer) -> None:
    module = RobotControlTendonsModule({})
    module.finalize(container)
    assert container.num_actions == 8  # 6 root + 2 tendons
    assert not container.robot.fixed_hand

    space = container.env.action_space
    assert space.shape == (8,)

    # Building again is a no-op (guard).
    RobotControlFloatingHandModule({}).finalize(container)
    assert container.env.action_space is space


def test_compute_modules_require_action_buffers(container: ModuleContainer) -> None:
    module = RobotControlTendonsModule({})
    module.finalize(container)
    container.act_buf = None
    container.scaled_act_buf = None
    with pytest.raises(RuntimeError, match="scaled_act_buf"):
        module.post_finalize(container)


# ---------------------------------------------------------------------------
# Apply modules
# ---------------------------------------------------------------------------
def test_apply_tendon_forces_creates_and_applies(container: ModuleContainer) -> None:
    module = ApplyTendonForcesModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.set_tendon_controls_buf[:] = 2.5
    module.step(container)

    assert container.set_tendon_controls_buf.shape == (1, 2)
    call = container.gym.calls[-1]
    assert call[0] == "set_spatial_tendon_forces"


def test_apply_motor_forces_adds_spring(container: ModuleContainer) -> None:
    module = ApplyMotorForcesModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.set_motor_cmd_buf = torch.full((1, 4), 99.0)
    container.motor_policy_force_buf = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    container.antagonistic_spring_force_buf = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    module.step(container)

    # Compose: zero + policy + spring.
    expected = torch.tensor([[0.6, 0.7, 0.8, 0.9]])
    assert torch.allclose(container.set_motor_cmd_buf, expected)
    assert container.gym.calls[-1][0] == "set_motor_forces"


def test_apply_motor_forces_without_policy_and_spring(container: ModuleContainer) -> None:
    module = ApplyMotorForcesModule({})
    module.finalize(container)
    module.post_finalize(container)

    # Tendon-hand wiring: neither policy nor spring buffer present -> zero.
    container.set_motor_cmd_buf = torch.full((1, 4), 99.0)
    container.gym.calls.clear()
    module.step(container)

    assert torch.all(container.set_motor_cmd_buf == 0.0)


def test_apply_root_velocity_requires_inverse_reset(container: ModuleContainer) -> None:
    module = ApplyRootVelocityModule({})
    module.finalize(container)
    container.inverse_reset_buf = None
    with pytest.raises(RuntimeError, match="inverse_reset_buf"):
        module.post_finalize(container)


def test_apply_root_velocity_creates_and_applies(container: ModuleContainer) -> None:
    module = ApplyRootVelocityModule({})
    module.finalize(container)
    module.post_finalize(container)

    assert container.set_root_transform_buf.shape == (1, 7)
    assert container.set_root_vel_buf.shape == (1, 6)
    assert container.set_joint_pos_buf.shape == (1, 0)
    module.step(container)
    assert container.gym.calls[-1][0] == "set_articulation_kinematic_states"


def test_apply_root_forces_creates_and_applies(container: ModuleContainer) -> None:
    module = ApplyRootForcesModule({})
    module.finalize(container)
    module.post_finalize(container)

    assert container.set_force_torque_buf.shape == (1, 1, 6)
    module.step(container)
    assert container.gym.calls[-1][0] == "set_link_external_forces"


# ---------------------------------------------------------------------------
# gravity_compensation / randomize_friction
# ---------------------------------------------------------------------------
def test_gravity_compensation_accepts_any_hand(container: ModuleContainer) -> None:
    # No fixed-hand gate: if the module is wired it is active.
    GravityCompensationModule({}).finalize(container)  # floating robot

    container.robot = _FakeRobot(fixed_hand=True, use_tendon=False)
    GravityCompensationModule({}).finalize(container)  # fixed robot


def test_gravity_compensation_requires_robot(container: ModuleContainer) -> None:
    container.robot = None
    with pytest.raises(RuntimeError, match="requires 'robot'"):
        GravityCompensationModule({}).finalize(container)


def test_gravity_compensation_step_writes_forces(container: ModuleContainer) -> None:
    container.robot = _FakeRobot(fixed_hand=True, use_tendon=False)
    container.robot.link_masses = torch.tensor([1.0])

    container.set_force_torque_buf = torch.zeros((1, 1, 6))
    GravityCompensationModule({}).step(container)

    assert torch.allclose(container.set_force_torque_buf[:, :, 2], torch.tensor([[9.81]]))


def test_randomize_friction_applies_coefficient(container: ModuleContainer) -> None:
    module = RandomizeFrictionModule({"friction_coefficient": 0.8})
    module.finalize(container)
    module.post_finalize(container)

    module.reset(container)

    assert container.set_static_friction_buf.shape == (1,)
    assert container.set_dynamic_friction_buf.shape == (1,)
    assert torch.allclose(container.set_static_friction_buf, torch.tensor([1.6]))
    assert torch.allclose(container.set_dynamic_friction_buf, torch.tensor([1.2]))
    assert container.gym.calls[-1][0] == "set_rigid_material_properties"


def test_randomize_friction_randomizes_multi_env_without_coefficient(container: ModuleContainer) -> None:
    module = RandomizeFrictionModule({})
    module.finalize(container)
    module.post_finalize(container)

    container.reset_buf = torch.zeros(4, dtype=torch.bool)
    module.reset(container)

    # Friction was randomized in [0.1, 1.0] then doubled.
    value = container.set_static_friction_buf[0].item()
    assert 0.2 <= value <= 2.0
