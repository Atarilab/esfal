import copy
import time
import torch
import pinocchio as pin
import mujoco

from numpy.core.multiarray import array as array

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import position_3d_callback
from robots.configs import Go2Config

from learning_jump_feasibility.data.dataset_supervised import get_dataloaders, transform_points

DATA_PATH = "/home/atari_ws/data/learning_jump_feasibility"

# Configuration and paths
cfg = Go2Config  # Assuming Go2Config is defined elsewhere

URDF_path, xml_string, package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir, models_path="./robots")

# Load robot
robot = QuadrupedWrapperAbstract(
    URDF_path,
    xml_string,
    package_dir,
    rotor_inertia=cfg.rotor_inertia,
    gear_ratio=cfg.gear_ratio,
    foot_size=cfg.foot_size,
)
robot.reset()

### Load data
_, test_dataloader = get_dataloaders("classifier", data_path=DATA_PATH, batch_size=32)
batch = next(iter(test_dataloader))

### Simulator
simulator = Simulator(robot, None)


with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
    
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    
    viewer.sync()
    sim_start_time = time.time()
    while (viewer.is_running()):
        
        for input in batch["input"]:
            
            state, vel, vj, current, target = torch.split(input, [16, 6, 12, 12, 12]) 
            xyz = torch.zeros(3)
            xyz[2] += - torch.min(current)
            q = torch.cat((xyz, state)).numpy()
            
            w_T_B = pin.XYZQUATToSE3(q[:7])
            positions_b = torch.cat((current, target)).reshape(-1, 3).numpy()
            positions_w = transform_points(w_T_B, positions_b)

            robot.mj_data.qpos = robot.pin2mj_state(copy.deepcopy(q))
            robot.step()

            position_3d_callback(viewer, positions_w)

            viewer.sync()
            time.sleep(2)

