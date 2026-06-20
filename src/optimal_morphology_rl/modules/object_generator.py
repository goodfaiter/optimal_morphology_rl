from typing import Dict, List, Optional
import torch
from abc import ABC, abstractmethod
import vlearn as v

from optimal_morphology_rl.helpers.numpy_vlearn import random_uniform_quaternion

import importlib.resources as resources

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_ASSET_ROOT = "/workspace/optimal_morphology_rl_assets/optimal_morphology_rl_assets/assets/objects"


class ObjectBase(ABC):
    """Abstract base class for objects in the environment."""

    def __init__(self, name: str):
        self.name = name
        self.handle = None

        # State buffer dictionaries (rigid: len=1, articulated: len=N)
        self.get_trans_object_to_world_buf: Optional[torch.Tensor] = None
        self.get_vel_in_world_buf: Optional[torch.Tensor] = None
        self.set_trans_object_to_world_buf: Optional[torch.Tensor] = None
        self.set_vel_in_world_buf: Optional[torch.Tensor] = None

        # GPU command
        self.get_kin_cmd = None
        self.set_kin_cmd = None
        self.gpu_get_object_kin_cmd_array = None
        self.gpu_set_object_kin_cmd_array = None

        # Goals
        self.goal_pos_in_world: Optional[torch.Tensor] = None
        self.goal_quat_object_to_world: Optional[torch.Tensor] = None

    @abstractmethod
    def load(self, env_def):
        """Load object into environment definition and return handle."""
        raise NotImplementedError

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        """Allocate GPU buffers shared by all object types (trans/vel/goal). Subclasses call super() then add their own."""
        self.get_trans_object_to_world_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.get_vel_in_world_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)
        self.set_trans_object_to_world_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.set_vel_in_world_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)
        self.goal_pos_in_world = torch.zeros((total_num_envs, 3), device=device, dtype=torch.float32)
        self.goal_quat_object_to_world = torch.zeros((total_num_envs, 4), device=device, dtype=torch.float32)

    @abstractmethod
    def refresh_buffers(self, gym: v.Gym):
        """Refresh state buffers from simulation."""
        raise NotImplementedError

    @abstractmethod
    def update_goal(self, reset_buf: torch.Tensor):
        """Update goal position and orientation for the object based on reset indices."""
        raise NotImplementedError

    @abstractmethod
    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        """Reset any object-specific buffers based on reset indices."""
        raise NotImplementedError

    @abstractmethod
    def get_link_offset(self) -> int:
        """Return offset of links contributed to the contact transform table."""
        raise NotImplementedError

    @abstractmethod
    def create_gpu_command(self, env_group, gym, reset_buf):
        """Create GPU command for reading object state."""
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

    def load(self, env_def):
        """Load object from file into environment definition."""
        env_def.import_definitions(
            self.asset_path,
            fixed=self.fixed,
            use_visual_mesh=False,
            force_mass_computation=False,
            force_inertia_computation=False,
        )

        object_root_trans_init = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0))

        object_def_handle = env_def.get_rigid_body_def_handle_by_name(self.name)
        self.handle = env_def.create_rigid_body(object_def_handle, object_root_trans_init, self.name)

    def create_gpu_command(self, env_group, gym, reset_buf):
        """Create GPU command for reading object state."""
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
        """Refresh state buffers from simulation."""
        gym.get_rigid_body_kinematic_states(self.gpu_get_object_kin_cmd_array)

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.2
        self.goal_quat_object_to_world[reset_buf, :] = random_uniform_quaternion(
            reset_buf.sum().item(), device=reset_buf.device, dtype=torch.float32
        )

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        """Reset any object-specific buffers based on reset indices."""
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[0.0, 0.0, 0.025]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        gym.set_rigid_body_kinematic_states(self.gpu_set_object_kin_cmd_array)

        self.update_goal(reset_buf)

    def get_link_offset(self) -> int:
        return 1


class LoadedArticulatedObject(ObjectBase):
    """Object loaded from a file (URDF/VSIM)."""

    def __init__(self, name: str, asset_path: str, fixed: bool = False):
        super().__init__(name)
        self.asset_path = asset_path
        self.fixed = fixed
        self.num_joints = 0
        self.num_links = 0
        self.link_names: List[str] = []

        self.get_joint_pos_buf: Optional[torch.Tensor] = None
        self.get_joint_vel_buf: Optional[torch.Tensor] = None
        self.set_joint_pos_buf: Optional[torch.Tensor] = None
        self.set_joint_vel_buf: Optional[torch.Tensor] = None

    def load(self, env_def):
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
        art_def = env_def.get_articulation_def(object_def_handle)
        self.num_joints = art_def.get_num_joint_dof_defs()
        self.num_links = art_def.get_num_link_defs()
        self.link_names = [art_def.get_link_def(i).name for i in range(self.num_links)]
        self.handle = env_def.create_articulation(object_def_handle, object_root_trans_init, self.name)

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        """Allocate GPU buffers for articulated state."""
        super().allocate_buffers(total_num_envs, device)

        self.get_joint_pos_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.get_joint_vel_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.set_joint_pos_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)
        self.set_joint_vel_buf = torch.zeros((total_num_envs, self.num_joints), device=device, dtype=torch.float32)

    def refresh_buffers(self, gym: v.Gym):
        """Refresh state buffers from simulation."""
        gym.get_articulation_kinematic_states(self.gpu_get_object_kin_cmd_array)

    def create_gpu_command(self, env_group, gym, reset_buf):
        """Create GPU command for reading articulated state."""
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


class Pen(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="pen", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/pen.vsim"))


class Tomato(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="tomato", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/tomato.vsim"))


class Knife(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="knife", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/kitchen_knife.vsim"))


class Mug(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="mug", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/mug.vsim"))


class Table(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="table", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/table.vsim"), fixed=True)

    @property
    def half_size_tensor(self) -> torch.Tensor:
        return torch.tensor([0.2, 0.3, 0.01], device=self.get_trans_object_to_world_buf.device, dtype=torch.float32)

    @property
    def half_size(self) -> torch.Tensor:
        return v.Vec3(0.2, 0.3, 0.01)


class TableWithCamera(LoadedRigidObject):
    def __init__(self):
        super().__init__(name="table_with_camera", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/table_with_camera.vsim"), fixed=True)

    @property
    def half_size_tensor(self) -> torch.Tensor:
        return torch.tensor([0.2, 0.3, 0.01], device=self.get_trans_object_to_world_buf.device, dtype=torch.float32)

    @property
    def half_size(self) -> torch.Tensor:
        return v.Vec3(0.2, 0.3, 0.01)


class Drawer(LoadedArticulatedObject):
    def __init__(self):
        super().__init__(name="drawer", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/drawer.vsim"), fixed=True)

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.0
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        """Reset any object-specific buffers based on reset indices."""
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[0.2, 0.0, 0.1]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0

        gym.set_articulation_kinematic_states(self.gpu_set_object_kin_cmd_array)

        self.update_goal(reset_buf)


class Button(LoadedArticulatedObject):
    def __init__(self):
        super().__init__(name="button", asset_path=str(resources.files("optimal_morphology_rl_assets.assets") / "objects/button.vsim"), fixed=True)

    def update_goal(self, reset_buf: torch.Tensor):
        self.goal_pos_in_world[reset_buf, 0] = 0.3
        self.goal_pos_in_world[reset_buf, 1] = 0.0
        self.goal_pos_in_world[reset_buf, 2] = 0.1
        self.goal_quat_object_to_world[reset_buf, :] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)

    def reset_idx(self, gym: v.Gym, reset_buf: torch.Tensor):
        """Reset any object-specific buffers based on reset indices."""
        self.set_trans_object_to_world_buf[reset_buf, :4] = torch.tensor(_IDENTITY_QUAT, device=reset_buf.device)
        self.set_trans_object_to_world_buf[reset_buf, 4:] = torch.tensor([[0.2, 0.0, 0.1]], device=reset_buf.device)
        self.set_vel_in_world_buf[reset_buf, :] = 0.0
        gym.set_articulation_kinematic_states(self.gpu_set_object_kin_cmd_array)

        self.update_goal(reset_buf)

    


class ObjectGenerator:
    """Container for all objects in the environment."""

    OBJECT_REGISTRY: Dict[str, type] = {
        "pen": Pen,
        "tomato": Tomato,
        "knife": Knife,
        "mug": Mug,
        "table": Table,
        "table_with_camera": TableWithCamera,
        "drawer": Drawer,
        "button": Button,
    }

    def __init__(self, object_names: List[str]):
        """
        Initialize ObjectGenerator.

        Args:
            object_names: List of object names to create (e.g., ["knife", "table"])
        """
        self.object_names = object_names

        # Create object instances
        self.objects: Dict[str, ObjectBase] = {}
        for obj_name in object_names:
            if obj_name not in self.OBJECT_REGISTRY:
                raise ValueError(f"Unknown object: {obj_name}. Available: {list(self.OBJECT_REGISTRY.keys())}")
            self.objects[obj_name] = self.OBJECT_REGISTRY[obj_name]()

    def load(self, env_def):
        """Load objects into environment definition."""
        for obj_name in self.object_names:
            self.objects[obj_name].load(env_def)

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        """Allocate GPU buffers for all objects."""
        for obj in self.objects.values():
            obj.allocate_buffers(total_num_envs, device)

    def create_gpu_commands(self, env_group, gym, reset_buf) -> any:
        """Create and return GPU command array for all objects."""
        for obj in self.objects.values():
            obj.create_gpu_command(env_group, gym, reset_buf)

    def refresh_buffers(self, gym):
        """Refresh state buffers for all objects."""
        for obj in self.objects.values():
            obj.refresh_buffers(gym)

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
