from typing import Dict, List
import numpy as np
from abc import ABC, abstractmethod
import torch
import vlearn as v


class ObjectBase(ABC):
    """Abstract base class for objects in the environment."""
    
    def __init__(self, name: str):
        self.name = name
        self.handle = None
        
        # State buffers
        self.get_pos_buf: torch.Tensor = None  # Read: quat(4) + pos(3)
        self.get_vel_buf: torch.Tensor = None  # Read: angular(3) + linear(3)
        self.set_pos_buf: torch.Tensor = None  # Write: quat(4) + pos(3)
        self.set_vel_buf: torch.Tensor = None  # Write: angular(3) + linear(3)
        
        # GPU command
        self.get_kin_cmd = None
        self.set_kin_cmd = None
        self.gpu_get_object_kin_cmd_array = None
        self.gpu_set_object_kin_cmd_array = None

    @abstractmethod
    def load(self, env_def):
        """Load object into environment definition and return handle."""
        raise NotImplementedError

    def allocate_buffers(self, total_num_envs: int, device: torch.device):
        """Allocate GPU buffers for state."""
        self.get_pos_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.get_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)
        self.set_pos_buf = torch.zeros((total_num_envs, 7), device=device, dtype=torch.float32)
        self.set_vel_buf = torch.zeros((total_num_envs, 6), device=device, dtype=torch.float32)

    def create_gpu_command(self, env_group, gym, reset_buf):
        """Create GPU command for reading object state."""
        self.get_kin_cmd = env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_pos_buf),
            v.wrap_gpu_buffer(self.get_vel_buf),
            self.handle,
        )
        self.gpu_get_object_kin_cmd_array = gym.create_gpu_array([self.get_kin_cmd])

        self.set_kin_cmd = env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_pos_buf),
            v.wrap_gpu_buffer(self.set_vel_buf),
            self.handle,
            masks_buffer=v.wrap_gpu_buffer(reset_buf),
        )
        self.gpu_set_object_kin_cmd_array = gym.create_gpu_array([self.set_kin_cmd])

    @property
    def pos_in_world(self) -> torch.Tensor:
        return self.get_pos_buf[:, 4:7]

    @property
    def quat_object_to_world(self) -> torch.Tensor:
        return self.get_pos_buf[:, 0:4]

    @property
    def linear_velocity_world(self) -> torch.Tensor:
        return self.get_vel_buf[:, 3:6]

    @property
    def angular_velocity_world(self) -> torch.Tensor:
        return self.get_vel_buf[:, :3]


class LoadedObject(ObjectBase):
    """Object loaded from a file (URDF/VSIM)."""
    
    def __init__(self, name: str, asset_path: str, use_visual_mesh: bool):
        super().__init__(name)
        self.asset_path = asset_path
        self.use_visual_mesh = use_visual_mesh

    def load(self, env_def):
        """Load object from file into environment definition."""
        env_def.import_definitions(
            self.asset_path,
            fixed=False,
            use_visual_mesh=self.use_visual_mesh,
            force_mass_computation=False,
            force_inertia_computation=False,
        )
                
        object_root_trans_init = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0.1))
        
        object_def_handle = env_def.get_rigid_body_def_handle_by_name(self.name)
        self.handle = env_def.create_rigid_body(object_def_handle, object_root_trans_init, self.name)


class CreatedObject(ObjectBase):
    """Object created programmatically (e.g., table)."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.rgb_material_handle = None


class Pen(LoadedObject):
    def __init__(self):
        super().__init__(
            name="pen",
            asset_path="/workspace/optimal_morphology_rl/assets/objects/pen_big.vsim",
            use_visual_mesh=False,
        )


class Tomato(LoadedObject):
    def __init__(self):
        super().__init__(
            name="tomato",
            asset_path="/workspace/optimal_morphology_rl/assets/objects/tomato.vsim",
            use_visual_mesh=False,
        )


class Knife(LoadedObject):
    def __init__(self):
        super().__init__(
            name="knife",
            asset_path="/workspace/optimal_morphology_rl/assets/objects/kitchen_knife.vsim",
            use_visual_mesh=True,
        )


class Mug(LoadedObject):
    def __init__(self):
        super().__init__(
            name="mug",
            asset_path="/workspace/optimal_morphology_rl/assets/objects/mug.vsim",
            use_visual_mesh=False,
        )


class SquareDonut(LoadedObject):
    def __init__(self):
        super().__init__(
            name="square_donut",
            asset_path="/workspace/optimal_morphology_rl/assets/objects/square_donut.vsim",
            use_visual_mesh=False,
        )


class Table(CreatedObject):
    """Table created as a box primitive."""
    
    def __init__(self):
        super().__init__(name="table")
        self.half_size = v.Vec3(0.2, 0.3, 0.01)

    def load(self, env_def):
        """Create table in environment definition."""
        # Create RGB material
        rgb_mat = v.RGBMaterial()
        rgb_mat.color = v.Vec3(1, 1, 0)
        rgb_mat.specular = 40
        rgb_mat.spec_intensity = 0.25
        self.rgb_material_handle = env_def.create_rgb_material(rgb_mat)

        # Create box
        table_def_handle = env_def.create_box_def(
            half_size=self.half_size, 
            name="table", 
            fixed=True, 
            rgb_material_handle=self.rgb_material_handle
        )
        self.handle = env_def.create_rigid_body(
            table_def_handle, 
            v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, -self.half_size.z)), 
            "table"
        )

    @property
    def half_size_tensor(self) -> torch.Tensor:
        return torch.tensor([self.half_size.x, self.half_size.y, self.half_size.z], device=self.device, dtype=torch.float32)


class ObjectGenerator:
    """Container for all objects in the environment."""
    
    OBJECT_REGISTRY: Dict[str, type] = {
        "pen": Pen,
        "tomato": Tomato,
        "knife": Knife,
        "mug": Mug,
        "square_donut": SquareDonut,
        "table": Table,
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

    def get_object(self, name: str) -> ObjectBase:
        """Get a specific object by name."""
        return self.objects.get(name)