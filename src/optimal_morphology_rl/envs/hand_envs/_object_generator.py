"""Legacy object generator used by the old hand-object environments."""

from typing import Dict, List, Optional
import math
import torch
from abc import ABC, abstractmethod
import vlearn as v

from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion

import importlib.resources as resources

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_ASSET_ROOT = "/workspace/optimal_morphology_rl_assets/optimal_morphology_rl_assets/assets/objects"


def _object_asset_path(name: str) -> str:
    return str(resources.files("optimal_morphology_rl_assets.assets") / f"objects/{name}.vsim")


class ObjectBase(ABC):
    """Abstract base class for objects in the environment."""

    def __init__(self, name: str):
        self.name = name
        self.handle = None

        self.get_trans_object_to_world_buf: Optional[torch.Tensor] = None
        self.get_vel_in_world_buf: Optional[torch.Tensor] = None
        self.set_trans_object_to_world_buf: Optional[torch.Tensor] = None
        self.set_vel_in_world_buf: Optional[torch.Tensor] = None

        self.get_kin_cmd = None
        self.set_kin_cmd = None
        self.gpu_get_object_kin_cmd_array = None
        self.gpu_set_object_kin_cmd_array = None

        self.goal_pos_in_world: Optional[torch.Tensor] = None
        self.goal_quat_object_to_world: Optional[torch.Tensor] = None

    def pre_physics_step(self, gym: v.Gym) -> None:
        pass

    @abstractmethod
    def load(self, env_def):
        raise NotImplementedError

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        self.get_trans_object_to_world_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.get_vel_in_world_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)
        self.set_trans_object_to_world_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.set_vel_in_world_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)
        self.goal_pos_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
        self.goal_quat_object_to_world = torch.zeros((total_num_envs, 4), device=device, dtype=torch.float32)

    @abstractmethod
    def refresh_buffers(self, gym: v.Gym):
        raise NotImplementedError

    @abstractmethod
    def update_goal(self, reset_buf: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_link_offset(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def create_gpu_command(self, env_group, gym, reset_buf):
        raise NotImplementedError

    @property
    def pos_in_world(self) -> torch.Tensor:
        return self.get_trans_object_to_world_buf[:, 4:7]

    @property
    def quat_object_to_world(self) -> torch.Tensor:
        return self.get_trans_object_to_world_buf[:, 0:4]

    @property
    def linear_velocity_world(self) -> torch.Tensor:
        return self.get_vel_in_world_buf[:, 3:6]

    @property
    def angular_velocity_world(self) -> torch.Tensor:
        return self.get_vel_in_world_buf[:, :3]

    @property
    def set_trans_object_to_world(self) -> torch.Tensor:
        return self.set_trans_object_to_world_buf

    @property
    def set_vel_in_world(self) -> torch.Tensor:
        return self.set_vel_in_world_buf


class LoadedRigidObject(ObjectBase):
    def __init__(self, name: str, asset_path: str, fixed: bool = False):
        super().__init__(name)
        self.asset_path = asset_path
        self.fixed = fixed

    def load(self, env_def):
        env_def.import_definitions(
            self.asset_path,
            fixed=self.fixed,
            use_visual_mesh=True,
            force_mass_computation=False,
            force_inertia_computation=False,
        )

        object_root_trans_init = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0))

        object_def_handle = env_def.get_rigid_body_def_handle_by_name(self.name)
        self.handle = env_def.create_rigid_body(object_def_handle, object_root_trans_init, self.name)

        rigid_mat = v.RigidMaterial()
        rigid_mat.static_friction = 0.0 if self.name not in ["table", "table_with_camera"] else 1.8
        rigid_mat.dynamic_friction = 0.0 if self.name not in ["table", "table_with_camera"] else 1.5
        rigid_mat.restitution = 0.0
        rigid_mat.damping = 0.0
        rigid_mat.roughness = 0.0
        rigid_mat_handle = env_def.create_rigid_material(rigid_mat)
        env_def.assign_rigid_material_to_rigid_body(object_def_handle, rigid_mat_handle)

    def create_gpu_command(self, env_group, gym, reset_buf):
        self.get_kin_cmd = env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_trans_object_to_world_buf),
            v.wrap_gpu_buffer(self.get_vel_in_world_buf),
            self.handle,
        )
        self.gpu_get_object_kin_cmd_array = gym.create_gpu_array([self.get_kin_cmd])

        self.set_kin_cmd = env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_trans_object_to_world_buf),
            v.wrap_gpu_buffer(self.set_vel_in_world_buf),
            self.handle,
            masks_buffer=v.wrap_gpu_buffer(reset_buf),
        )
        self.gpu_set_object_kin_cmd_array = gym.create_gpu_array([self.set_kin_cmd])

    def refresh_buffers(self, gym: v.Gym):
        gym.get_rigid_body_kinematic_states(self.gpu_get_object_kin_cmd_array)

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.2
        self.goal_quat_object_to_world[reset_buf, :] = random_uniform_quaternion(
            reset_buf.sum().item(), device=reset_buf.device, dtype=torch.float32
        )

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[0.0, 0.0, 0.025]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        gym.set_rigid_body_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)

    def get_link_offset(self) -> int:
        return 1


class LoadedArticulatedObject(ObjectBase):
    def __init__(self, name: str, asset_path: str, fixed: bool = False):
        super().__init__(name)
        self.asset_path = asset_path
        self.fixed = fixed
        self.art_def = None
        self.num_joints = 0
        self.num_links = 0
        self.num_motors = 0
        self.link_names: List[str] = []

        self.get_joint_pos_buf: Optional[torch.Tensor] = None
        self.get_joint_vel_buf: Optional[torch.Tensor] = None
        self.set_joint_pos_buf: Optional[torch.Tensor] = None
        self.set_joint_vel_buf: Optional[torch.Tensor] = None

    def load(self, env_def):
        env_def.import_definitions(
            self.asset_path,
            fixed=self.fixed,
            merge_fixed_joints=False,
            use_visual_mesh=False,
            force_mass_computation=False,
            force_inertia_computation=False,
        )

        object_root_trans_init = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0))

        object_def_handle = env_def.get_articulation_def_handle_by_name(self.name)
        self.art_def = env_def.get_articulation_def(object_def_handle)
        self.num_joints = self.art_def.get_num_joint_dof_defs()
        self.num_links = self.art_def.get_num_link_defs()
        self.num_motors = self.art_def.get_num_motor_defs()
        for i in range(self.num_links):
            print(self.art_def.get_link_def(i))
        self.link_names = [self.art_def.get_link_def(i).name for i in range(self.num_links)]
        self.handle = env_def.create_articulation(object_def_handle, object_root_trans_init, self.name)

        rigid_mat = v.RigidMaterial()
        rigid_mat.static_friction = 0.0
        rigid_mat.dynamic_friction = 0.0
        rigid_mat.restitution = 0.0
        rigid_mat.damping = 0.0
        rigid_mat.roughness = 0.0
        rigid_mat_handle = env_def.create_rigid_material(rigid_mat)
        for i in range(self.art_def.get_num_link_defs()):
            env_def.assign_rigid_material_to_articulation_link(object_def_handle, rigid_mat_handle, i)

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        super().allocate_buffers(total_num_envs, device)
        self.get_joint_pos_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.get_joint_vel_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.set_joint_pos_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.set_joint_vel_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)

    def refresh_buffers(self, gym: v.Gym):
        gym.get_articulation_kinematic_states(self.gpu_get_object_kin_cmd_array)

    def create_gpu_command(self, env_group, gym, reset_buf):
        get_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_joint_pos_buf),
            v.wrap_gpu_buffer(self.get_joint_vel_buf),
            v.wrap_gpu_buffer(self.get_trans_object_to_world_buf),
            v.wrap_gpu_buffer(self.get_vel_in_world_buf),
            self.handle,
            (0, self.num_joints),
            (0, 1),
        )
        self.gpu_get_object_kin_cmd_array = gym.create_gpu_array([get_kin_cmd])

        set_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_joint_pos_buf),
            v.wrap_gpu_buffer(self.set_joint_vel_buf),
            v.wrap_gpu_buffer(self.set_trans_object_to_world_buf),
            v.wrap_gpu_buffer(self.set_vel_in_world_buf),
            self.handle,
            (0, self.num_joints),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(reset_buf),
        )
        self.gpu_set_object_kin_cmd_array = gym.create_gpu_array([set_kin_cmd])

    def get_link_offset(self) -> int:
        return self.num_links


class Cube(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="cube", asset_path=_object_asset_path("cube_mid"))

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = -0.15
        self.goal_pos_in_world[reset_buf, 2] = 0.25
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor([0.7071068, -0.7071068, 0, 0], device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[-0.05, -0.15, 0.15]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        gym.set_rigid_body_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)


class Tomato(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="tomato", asset_path=_object_asset_path("tomato"))


class TomatoExtreme(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="tomato_extreme", asset_path=_object_asset_path("tomato_extreme"))

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[0.0, 0.0, 0.075]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        gym.set_rigid_body_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)


class Knife(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="knife", asset_path=_object_asset_path("kitchen_knife"))


class Mug(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="mug", asset_path=_object_asset_path("mug"))


def _assign_teal_material(env_def, object_name: str) -> None:
    teal_mat = v.RGBMaterial()
    teal_mat.color = v.Vec3(0.0, 0.5, 0.5)
    teal_mat.specular = 40.0
    teal_mat.spec_intensity = 0.25
    teal_mat_handle = env_def.create_rgb_material(teal_mat)
    rigid_body_def_handle = env_def.get_rigid_body_def_handle_by_name(object_name)
    env_def.assign_rgb_material_to_rigid_body(rigid_body_def_handle, teal_mat_handle)


class Table(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="table", asset_path=_object_asset_path("table"), fixed=True)

    def load(self, env_def):
        super().load(env_def)
        _assign_teal_material(env_def, self.name)

    @property
    def half_size_tensor(self) -> torch.Tensor:
        return torch.tensor([0.2, 0.3, 0.01], device=self.get_trans_object_to_world_buf.device, dtype=torch.float32)

    @property
    def half_size(self) -> v.Vec3:
        return v.Vec3(0.2, 0.3, 0.01)


class TableWithCamera(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="table_with_camera", asset_path=_object_asset_path("table_with_camera"), fixed=True)

    def load(self, env_def):
        super().load(env_def)
        _assign_teal_material(env_def, self.name)

    @property
    def half_size_tensor(self) -> torch.Tensor:
        return torch.tensor([0.2, 0.3, 0.01], device=self.get_trans_object_to_world_buf.device, dtype=torch.float32)

    @property
    def half_size(self) -> v.Vec3:
        return v.Vec3(0.2, 0.3, 0.01)


class Drawer(LoadedArticulatedObject):
    def __init__(
        self,
        spring_stiffness: float = 0.01,
        spring_damping: float = 0.05,
        spring_rest_angle: float = 0.0,
        max_spring_torque: float = 1.0,
        lock_stiffness: float = 100.0,
        lock_damping: float = 10.0,
        max_lock_force: float = 5.0,
        unlock_angle_threshold: float = math.radians(60.0),
    ):
        super().__init__(name="drawer", asset_path=_object_asset_path("drawer"), fixed=True)
        self.spring_stiffness = spring_stiffness
        self.spring_damping = spring_damping
        self.spring_rest_angle = spring_rest_angle
        self.max_spring_torque = max_spring_torque

        self.lock_stiffness = lock_stiffness
        self.lock_damping = lock_damping
        self.max_lock_force = max_lock_force
        self.unlock_angle_threshold = unlock_angle_threshold

        self.handle_joint_motor_index: Optional[int] = None
        self.handle_joint_dof_index: Optional[int] = None
        self.drawer_joint_motor_index: Optional[int] = None
        self.drawer_joint_dof_index: Optional[int] = None
        self.spring_motor_cmd_buf: Optional[torch.Tensor] = None
        self.gpu_spring_motor_cmd_array = None
        self.unlocked_buf: Optional[torch.Tensor] = None

    def load(self, env_def):
        super().load(env_def)

        if self.art_def is None:
            return

        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        for i in range(self.art_def.get_num_motor_defs()):
            motor_def = self.art_def.get_motor_def(i)
            if motor_def.joint_name == "handle_joint":
                self.handle_joint_motor_index = i
                self.handle_joint_dof_index = motor_def.dof_index
            elif motor_def.joint_name == "drawer_joint":
                self.drawer_joint_motor_index = i
                self.drawer_joint_dof_index = motor_def.dof_index

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        super().allocate_buffers(total_num_envs, device)
        if self.handle_joint_motor_index is not None and self.drawer_joint_motor_index is not None and self.num_motors > 0:
            self.spring_motor_cmd_buf = torch.zeros((total_num_envs, self.num_motors), device=device, dtype=torch.float32)
            self.unlocked_buf = torch.zeros((total_num_envs,), device=device, dtype=torch.bool)

    def create_gpu_command(self, env_group, gym, reset_buf):
        super().create_gpu_command(env_group, gym, reset_buf)
        if self.spring_motor_cmd_buf is not None and self.num_motors > 0:
            set_motor_cmd = env_group.create_motor_control_command(
                v.wrap_gpu_buffer(self.spring_motor_cmd_buf),
                self.handle,
                (0, self.num_motors),
            )
            self.gpu_spring_motor_cmd_array = gym.create_gpu_array([set_motor_cmd])

    def pre_physics_step(self, gym: v.Gym):
        if (
            self.handle_joint_dof_index is None
            or self.handle_joint_motor_index is None
            or self.drawer_joint_dof_index is None
            or self.drawer_joint_motor_index is None
            or self.spring_motor_cmd_buf is None
            or self.gpu_spring_motor_cmd_array is None
            or self.unlocked_buf is None
        ):
            return

        q_handle = self.get_joint_pos_buf[:, self.handle_joint_dof_index]
        qd_handle = self.get_joint_vel_buf[:, self.handle_joint_dof_index]
        torque = -self.spring_stiffness * (q_handle - self.spring_rest_angle)
        torque = torch.clamp(torque, -self.max_spring_torque, self.max_spring_torque)

        self.unlocked_buf |= torch.abs(q_handle) >= self.unlock_angle_threshold

        q_drawer = self.get_joint_pos_buf[:, self.drawer_joint_dof_index]
        qd_drawer = self.get_joint_vel_buf[:, self.drawer_joint_dof_index]
        lock_force = torch.zeros_like(q_drawer)
        locked = ~self.unlocked_buf
        if locked.any():
            lock_force[locked] = -self.lock_stiffness * q_drawer[locked] - self.lock_damping * qd_drawer[locked]
            lock_force = torch.clamp(lock_force, -self.max_lock_force, self.max_lock_force)

        self.spring_motor_cmd_buf.zero_()
        self.spring_motor_cmd_buf[:, self.handle_joint_motor_index] = torque
        self.spring_motor_cmd_buf[:, self.drawer_joint_motor_index] = lock_force
        gym.set_motor_forces(self.gpu_spring_motor_cmd_array)

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        half = math.sin(math.pi / 4.0)
        w = math.cos(math.pi / 4.0)
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor([half, 0.0, 0.0, w], device=reset_buf.device)

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[0.2, 0.0, 0.1]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0

        if self.unlocked_buf is not None:
            self.unlocked_buf[reset_buf] = False

        gym.set_articulation_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)


class Button(LoadedArticulatedObject):
    def __init__(
        self,
        spring_stiffness: float = 0.1,
        spring_damping: float = 0.01,
        spring_rest_position: float = 0.0,
        max_spring_force: float = 5.0,
    ):
        super().__init__(name="button", asset_path=_object_asset_path("button"), fixed=True)
        self.spring_stiffness = spring_stiffness
        self.spring_damping = spring_damping
        self.spring_rest_position = spring_rest_position
        self.max_spring_force = max_spring_force

        self.button_joint_motor_index: Optional[int] = None
        self.button_joint_dof_index: Optional[int] = None
        self.spring_motor_cmd_buf: Optional[torch.Tensor] = None
        self.gpu_spring_motor_cmd_array = None

    def load(self, env_def):
        super().load(env_def)

        if self.art_def is None:
            return

        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        for i in range(self.art_def.get_num_motor_defs()):
            motor_def = self.art_def.get_motor_def(i)
            if motor_def.joint_name == "button":
                self.button_joint_motor_index = i
                self.button_joint_dof_index = motor_def.dof_index

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        super().allocate_buffers(total_num_envs, device)
        if self.button_joint_motor_index is not None and self.num_motors > 0:
            self.spring_motor_cmd_buf = torch.zeros((total_num_envs, self.num_motors), device=device, dtype=torch.float32)

    def create_gpu_command(self, env_group, gym, reset_buf):
        super().create_gpu_command(env_group, gym, reset_buf)
        if self.spring_motor_cmd_buf is not None and self.num_motors > 0:
            set_motor_cmd = env_group.create_motor_control_command(
                v.wrap_gpu_buffer(self.spring_motor_cmd_buf),
                self.handle,
                (0, self.num_motors),
            )
            self.gpu_spring_motor_cmd_array = gym.create_gpu_array([set_motor_cmd])

    def pre_physics_step(self, gym: v.Gym):
        if (
            self.button_joint_dof_index is None
            or self.button_joint_motor_index is None
            or self.spring_motor_cmd_buf is None
            or self.gpu_spring_motor_cmd_array is None
        ):
            return

        q_button = self.get_joint_pos_buf[:, self.button_joint_dof_index]
        qd_button = self.get_joint_vel_buf[:, self.button_joint_dof_index]
        force = -self.spring_stiffness * (q_button - self.spring_rest_position)
        force = force - self.spring_damping * qd_button
        force = torch.clamp(force, -self.max_spring_force, self.max_spring_force)

        self.spring_motor_cmd_buf.zero_()
        self.spring_motor_cmd_buf[:, self.button_joint_motor_index] = force
        gym.set_motor_forces(self.gpu_spring_motor_cmd_array)

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.3
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[0.2, 0.0, 0.1]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0

        if self.button_joint_dof_index is not None:
            self.set_joint_pos_buf[reset_buf, self.button_joint_dof_index] = self.spring_rest_position
            self.set_joint_vel_buf[reset_buf, self.button_joint_dof_index] = 0.0

        gym.set_articulation_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)


class ButtonDifficult(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "button_difficult"
        self.asset_path = _object_asset_path("button_difficult")

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.35
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)


class ObjectGenerator:
    """Container for all objects in the environment."""

    OBJECT_REGISTRY: Dict[str, type] = {
        "cube": Cube,
        "tomato": Tomato,
        "tomato_extreme": TomatoExtreme,
        "knife": Knife,
        "mug": Mug,
        "table": Table,
        "table_with_camera": TableWithCamera,
        "drawer": Drawer,
        "button": Button,
        "button_difficult": ButtonDifficult,
    }

    def __init__(self, object_names: List[str]):
        self.object_names = object_names
        self.objects: Dict[str, ObjectBase] = {}
        for obj_name in object_names:
            if obj_name not in self.OBJECT_REGISTRY:
                raise ValueError(f"Unknown object: {obj_name}. Available: {list(self.OBJECT_REGISTRY.keys())}")
            self.objects[obj_name] = self.OBJECT_REGISTRY[obj_name]()

    def load(self, env_def):
        for obj_name in self.object_names:
            self.objects[obj_name].load(env_def)

    def allocate_buffers(self, total_num_envs: int, device):
        for obj in self.objects.values():
            obj.allocate_buffers(total_num_envs, device)

    def create_gpu_commands(self, env_group, gym, reset_buf) -> any:
        for obj in self.objects.values():
            obj.create_gpu_command(env_group, gym, reset_buf)

    def refresh_buffers(self, gym):
        for obj in self.objects.values():
            obj.refresh_buffers(gym)

    def post_physics_step(self, gym) -> None:
        """Legacy no-op hook for the old environment loop."""
        pass

    def pre_physics_step(self, gym) -> None:
        for obj in self.objects.values():
            obj.pre_physics_step(gym)

    def reset_idx(self, gym, reset_buf) -> None:
        if reset_buf.sum() == 0:
            return
        for obj in self.objects.values():
            obj.reset_idx(gym, reset_buf)

    def get_object(self, name: str) -> ObjectBase:
        return self.objects.get(name)

    def get_object_link_offset(self, name: str) -> int:
        offset = 0
        for obj_name in self.object_names:
            offset += self.objects[obj_name].get_link_offset()
            if obj_name == name:
                return offset
        raise ValueError(f"Unknown object: {name}.")
