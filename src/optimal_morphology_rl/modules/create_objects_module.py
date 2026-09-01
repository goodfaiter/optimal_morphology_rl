"""Module that creates scene objects and exposes them on the shared container."""

from __future__ import annotations

import importlib.resources as resources
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import vlearn as v

from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion
from optimal_morphology_rl.modules.base_module import BaseModule
from optimal_morphology_rl.modules.module_container import ModuleContainer
from optimal_morphology_rl.modules.module_manager import register_module


_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _object_asset_path(name: str) -> str:
    return str(resources.files("optimal_morphology_rl_assets.assets") / f"objects/{name}.vsim")


class ObjectBase(ABC):
    """Abstract base class for objects in the environment."""

    def __init__(self, name: str):
        self.name = name
        self.handle: Any = None

        # State buffers attached by update/create_objects modules.
        self.get_trans_object_to_world_buf: Optional[torch.Tensor] = None
        self.get_vel_in_world_buf: Optional[torch.Tensor] = None
        self.set_trans_object_to_world_buf: Optional[torch.Tensor] = None
        self.set_vel_in_world_buf: Optional[torch.Tensor] = None

        # GPU command arrays attached by update/create_objects modules.
        self.gpu_get_object_kin_cmd_array: Any = None
        self.gpu_set_object_kin_cmd_array: Any = None

        # Goal buffers attached by create_objects module.
        self.goal_pos_in_world: Optional[torch.Tensor] = None
        self.goal_quat_object_to_world: Optional[torch.Tensor] = None

    @abstractmethod
    def load(self, env_def) -> None:
        """Load object into environment definition and set self.handle."""
        raise NotImplementedError

    @abstractmethod
    def update_goal(self, reset_buf: torch.Tensor) -> None:
        """Update goal position and orientation for the object."""
        raise NotImplementedError

    @abstractmethod
    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        """Reset object state for the given reset indices."""
        raise NotImplementedError

    @abstractmethod
    def get_link_offset(self) -> int:
        """Return offset of links contributed to the contact transform table."""
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
    """Object loaded from a file (URDF/VSIM)."""

    def __init__(self, name: str, asset_path: str, fixed: bool = False):
        super().__init__(name)
        self.asset_path = asset_path
        self.fixed = fixed

    def load(self, env_def) -> None:
        """Load object from file into environment definition."""
        env_def.import_definitions(
            self.asset_path,
            fixed=self.fixed,
            use_visual_mesh=True,
            force_mass_computation=False,
            force_inertia_computation=False,
        )

        object_root_trans_init = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0))

        object_def_handle = env_def.get_rigid_body_def_handle_by_name(self.name)
        self.handle = env_def.create_rigid_body(
            object_def_handle, object_root_trans_init, self.name
        )

        # The friction is average between two objects. Set this one to 0 and the
        # robot hand to desired * 2.
        rigid_mat = v.RigidMaterial()
        rigid_mat.static_friction = 0.0 if self.name not in ["table", "table_with_camera"] else 1.8
        rigid_mat.dynamic_friction = 0.0 if self.name not in ["table", "table_with_camera"] else 1.5
        rigid_mat.restitution = 0.0
        rigid_mat.damping = 0.0
        rigid_mat.roughness = 0.0
        rigid_mat_handle = env_def.create_rigid_material(rigid_mat)
        env_def.assign_rigid_material_to_rigid_body(object_def_handle, rigid_mat_handle)

    def update_goal(self, reset_buf: torch.Tensor) -> None:
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.2
        self.goal_quat_object_to_world[reset_buf, :] = random_uniform_quaternion(
            reset_buf.sum().item(), device=reset_buf.device, dtype=torch.float32
        )

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor(
            [[0.0, 0.0, 0.025]], device=reset_buf.device
        )
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        gym.set_rigid_body_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)

    def get_link_offset(self) -> int:
        return 1


class LoadedArticulatedObject(ObjectBase):
    """Articulated object loaded from a file (URDF/VSIM)."""

    def __init__(self, name: str, asset_path: str, fixed: bool = False):
        super().__init__(name)
        self.asset_path = asset_path
        self.fixed = fixed
        self.art_def: Any = None
        self.num_joints: int = 0
        self.num_links: int = 0
        self.num_motors: int = 0
        self.link_names: List[str] = []

        # Joint state buffers attached by create_objects/update_objects modules.
        self.get_joint_pos_buf: Optional[torch.Tensor] = None
        self.get_joint_vel_buf: Optional[torch.Tensor] = None
        self.set_joint_pos_buf: Optional[torch.Tensor] = None
        self.set_joint_vel_buf: Optional[torch.Tensor] = None

    def load(self, env_def) -> None:
        """Load articulated object from file into environment definition."""
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
        self.handle = env_def.create_articulation(
            object_def_handle, object_root_trans_init, self.name
        )

        rigid_mat = v.RigidMaterial()
        rigid_mat.static_friction = 0.0
        rigid_mat.dynamic_friction = 0.0
        rigid_mat.restitution = 0.0
        rigid_mat.damping = 0.0
        rigid_mat.roughness = 0.0
        rigid_mat_handle = env_def.create_rigid_material(rigid_mat)
        for i in range(self.art_def.get_num_link_defs()):
            env_def.assign_rigid_material_to_articulation_link(
                object_def_handle, rigid_mat_handle, i
            )

    def update_goal(self, reset_buf: torch.Tensor) -> None:
        raise NotImplementedError

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor(
            [[0.2, 0.0, 0.1]], device=reset_buf.device
        )
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        self.set_joint_pos_buf[reset_buf, :] = 0.0
        self.set_joint_vel_buf[reset_buf, :] = 0.0
        gym.set_articulation_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)

    def get_link_offset(self) -> int:
        return self.num_links


class Cube(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="cube", asset_path=_object_asset_path("cube_mid"))

    def update_goal(self, reset_buf: torch.Tensor) -> None:
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = -0.15
        self.goal_pos_in_world[reset_buf, 2] = 0.25
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(
            [0.7071068, -0.7071068, 0, 0], device=reset_buf.device
        )
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor(
            [[-0.05, -0.15, 0.15]], device=reset_buf.device
        )
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        gym.set_rigid_body_kinematic_states(self.gpu_set_object_kin_cmd_array)
        self.update_goal(reset_buf)


class Tomato(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="tomato", asset_path=_object_asset_path("tomato"))


class TomatoExtreme(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="tomato_extreme", asset_path=_object_asset_path("tomato_extreme"))

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor(
            [[0.0, 0.0, 0.075]], device=reset_buf.device
        )
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
    """Assign a teal RGB material to a rigid body for nicer rendering."""
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

    def load(self, env_def) -> None:
        super().load(env_def)
        _assign_teal_material(env_def, self.name)

    @property
    def half_size_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [0.2, 0.3, 0.01],
            device=self.get_trans_object_to_world_buf.device,
            dtype=torch.float32,
        )

    @property
    def half_size(self) -> v.Vec3:
        return v.Vec3(0.2, 0.3, 0.01)


class TableWithCamera(LoadedRigidObject):
    def __init__(self):
        super().__init__(
            name="table_with_camera",
            asset_path=_object_asset_path("table_with_camera"),
            fixed=True,
        )

    def load(self, env_def) -> None:
        super().load(env_def)
        _assign_teal_material(env_def, self.name)

    @property
    def half_size_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [0.2, 0.3, 0.01],
            device=self.get_trans_object_to_world_buf.device,
            dtype=torch.float32,
        )

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

        # Set by load() once the articulation definition is available.
        self.handle_joint_motor_index: Optional[int] = None
        self.handle_joint_dof_index: Optional[int] = None
        self.drawer_joint_motor_index: Optional[int] = None
        self.drawer_joint_dof_index: Optional[int] = None

        # Control buffers attached by object_control module.
        self.spring_motor_cmd_buf: Optional[torch.Tensor] = None
        self.gpu_spring_motor_cmd_array: Any = None
        self.unlocked_buf: Optional[torch.Tensor] = None

    def load(self, env_def) -> None:
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

    def update_goal(self, reset_buf: torch.Tensor) -> None:
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        half = math.sin(math.pi / 4.0)
        w = math.cos(math.pi / 4.0)
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(
            [half, 0.0, 0.0, w], device=reset_buf.device
        )

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor(
            [[0.2, 0.0, 0.1]], device=reset_buf.device
        )
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

        # Set by load() once the articulation definition is available.
        self.button_joint_motor_index: Optional[int] = None
        self.button_joint_dof_index: Optional[int] = None

        # Control buffers attached by object_control module.
        self.spring_motor_cmd_buf: Optional[torch.Tensor] = None
        self.gpu_spring_motor_cmd_array: Any = None

    def load(self, env_def) -> None:
        super().load(env_def)

        if self.art_def is None:
            return

        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        for i in range(self.art_def.get_num_motor_defs()):
            motor_def = self.art_def.get_motor_def(i)
            if motor_def.joint_name == "button":
                self.button_joint_motor_index = i
                self.button_joint_dof_index = motor_def.dof_index

    def update_goal(self, reset_buf: torch.Tensor) -> None:
        self.goal_pos_in_world[reset_buf, 0] = 0.3
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor(
            [[0.2, 0.0, 0.1]], device=reset_buf.device
        )
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

    def update_goal(self, reset_buf: torch.Tensor) -> None:
        self.goal_pos_in_world[reset_buf, 0] = 0.35
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(
            _IDENTITY_QUAT, device=reset_buf.device
        )


#: Mapping from object name to object class.
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


class ObjectGenerator:
    """Container for all objects in the environment."""

    def __init__(self, object_names: List[str]):
        self.object_names = object_names
        self.objects: Dict[str, ObjectBase] = {}
        for obj_name in object_names:
            if obj_name not in OBJECT_REGISTRY:
                raise ValueError(
                    f"Unknown object: {obj_name}. Available: {list(OBJECT_REGISTRY.keys())}"
                )
            self.objects[obj_name] = OBJECT_REGISTRY[obj_name]()

    def load(self, env_def) -> None:
        """Load objects into environment definition."""
        for obj in self.objects.values():
            obj.load(env_def)

    def get_object(self, name: str) -> ObjectBase:
        """Get a specific object by name."""
        return self.objects.get(name)

    def get_object_link_offset(self, name: str) -> int:
        """Return link-based offset for the object based on object order."""
        offset = 0
        for obj_name in self.object_names:
            offset += self.objects[obj_name].get_link_offset()
            if obj_name == name:
                return offset
        raise ValueError(f"Unknown object: {name}.")

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor) -> None:
        """Reset objects selected by the reset buffer."""
        if reset_buf.sum() == 0:
            return
        for obj in self.objects.values():
            obj.reset_idx(gym, reset_buf)


# ---------------------------------------------------------------------------
# Set-buffer helpers used by CreateObjectsModule.
# ---------------------------------------------------------------------------
def _allocate_set_buffers(obj: ObjectBase, total_num_envs: int, device: torch.device) -> None:
    """Allocate buffers used to reset/write object state."""
    obj.set_trans_object_to_world_buf = torch.zeros(
        (total_num_envs, 7), device=device, dtype=torch.float32
    )
    obj.set_vel_in_world_buf = torch.zeros(
        (total_num_envs, 6), device=device, dtype=torch.float32
    )
    obj.goal_pos_in_world = torch.zeros(
        (total_num_envs, 3), device=device, dtype=torch.float32
    )
    obj.goal_quat_object_to_world = torch.zeros(
        (total_num_envs, 4), device=device, dtype=torch.float32
    )

    if isinstance(obj, LoadedArticulatedObject):
        obj.set_joint_pos_buf = torch.zeros(
            (total_num_envs, obj.num_joints), device=device, dtype=torch.float32
        )
        obj.set_joint_vel_buf = torch.zeros(
            (total_num_envs, obj.num_joints), device=device, dtype=torch.float32
        )


def _create_set_gpu_commands(obj: ObjectBase, env_group: Any, gym: v.Gym, reset_buf: torch.Tensor) -> None:
    """Create GPU commands used to reset object state."""
    if isinstance(obj, LoadedRigidObject):
        set_kin_cmd = env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(obj.set_trans_object_to_world_buf),
            v.wrap_gpu_buffer(obj.set_vel_in_world_buf),
            obj.handle,
            masks_buffer=v.wrap_gpu_buffer(reset_buf),
        )
        obj.gpu_set_object_kin_cmd_array = gym.create_gpu_array([set_kin_cmd])
    elif isinstance(obj, LoadedArticulatedObject):
        set_kin_cmd = env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(obj.set_joint_pos_buf),
            v.wrap_gpu_buffer(obj.set_joint_vel_buf),
            v.wrap_gpu_buffer(obj.set_trans_object_to_world_buf),
            v.wrap_gpu_buffer(obj.set_vel_in_world_buf),
            obj.handle,
            (0, obj.num_joints),
            (0, 1),
            masks_buffer=v.wrap_gpu_buffer(reset_buf),
        )
        obj.gpu_set_object_kin_cmd_array = gym.create_gpu_array([set_kin_cmd])


@register_module("create_objects")
class CreateObjectsModule(BaseModule):
    """Loads scene objects, manages their GPU state, and exposes them on the container."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.reward_object_name: str = self.config.get("reward_object", "drawer")
        self.scene_objects: List[str] = list(self.config.get("scene_objects", []))
        self.record_output_path = self.config.get("record_output_path", None)
        self.generator: ObjectGenerator | None = None

    def finalize(self, container: ModuleContainer) -> None:
        """Instantiate and load requested objects into the environment definition."""
        if container.get("env_def") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'env_def' in the shared container. "
                "Ensure create_rigid_vsim_envs is listed before create_objects."
            )

        table_name = "table" if self.record_output_path is None else "table_with_camera"

        object_names: List[str] = [self.reward_object_name]
        for obj in self.scene_objects:
            if obj not in object_names:
                object_names.append(obj)
        if table_name not in object_names:
            object_names.append(table_name)

        self.generator = ObjectGenerator(object_names)
        self.generator.load(container.env_def)

        container.objects = self.generator.objects
        container.create_objects = self
        container.reward_object_name = self.reward_object_name
        container.reward_object = self.generator.get_object(self.reward_object_name)
        container.object_names = object_names

    def post_finalize(self, container: ModuleContainer) -> None:
        """Allocate set buffers and create set GPU commands."""
        if container.get("env_group") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'env_group' in the shared container."
            )
        if container.get("total_num_envs") is None or container.get("device") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'total_num_envs' and 'device' in the shared container."
            )
        if container.get("reset_buf") is None:
            raise RuntimeError(
                "CreateObjectsModule requires 'reset_buf' in the shared container."
            )

        total_num_envs = container.total_num_envs
        device = container.device
        reset_buf = container.reset_buf

        for obj in self.generator.objects.values():
            _allocate_set_buffers(obj, total_num_envs, device)
            _create_set_gpu_commands(obj, container.env_group, container.gym, reset_buf)

        container.object_link_offsets = {
            name: self.generator.get_object_link_offset(name)
            for name in self.generator.object_names
        }
        container.reward_object_link_offset = self.generator.get_object_link_offset(
            self.reward_object_name
        )

    def reset(self, container: ModuleContainer) -> None:
        """Reset objects selected by the environment's reset buffer."""
        if self.generator is None:
            return
        reset_buf = container.get("reset_buf")
        if reset_buf is None or reset_buf.sum() == 0:
            return
        self.generator.reset_idx(container.gym, reset_buf)

    def get_object(self, name: str) -> ObjectBase:
        """Get a specific object by name."""
        if self.generator is None:
            raise RuntimeError("CreateObjectsModule has not been finalized.")
        return self.generator.get_object(name)

    def get_object_link_offset(self, name: str) -> int:
        """Return link-based offset for the object based on object order."""
        if self.generator is None:
            raise RuntimeError("CreateObjectsModule has not been finalized.")
        return self.generator.get_object_link_offset(name)
