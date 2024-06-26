import copy
import glob
import time
import pinocchio as pin
from multiprocessing import Manager, Pool, Queue
import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
from mujoco._structs import MjData
import mujoco

from numpy.core.multiarray import array as array

from mpc_controller.motions.cyclic.go2_jump import jump
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract

from environment.stepping_stones import SteppingStonesEnv
from environment.sim import SteppingStonesSimulator
from utils.visuals import position_3d_callback
from robots.configs import Go2Config

from learning_jump_feasibility.test.test_utils import load_model
from learning_jump_feasibility.test.test_utils import predict_next_state, is_feasible

RUN_ID = 10
REGRESSOR_PATH = f"/home/atari_ws/project/learning_jump_feasibility/logs/MLP_regressor/{RUN_ID}/MLP.pth"

RUN_ID = 1
CLASSIFIER_PATH = f"/home/atari_ws/project/learning_jump_feasibility/logs/MLPclassifierBinary/{RUN_ID}/MLP.pth"

### Configuration and paths
cfg = Go2Config  # Assuming Go2Config is defined elsewhere
URDF_path, xml_string, package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir, models_path="./robots")

### Stepping stones env
stepping_stones_height = 0.1
stepping_stones = SteppingStonesEnv(
    grid_size=(10, 3),
    spacing=(0.18, 0.14),
    size_ratio=(0.8, 0.8),
    height=stepping_stones_height,
    randomize_pos_ratio=0.,
    randomize_size_ratio=[0.55, 0.55]
)

xml_string = stepping_stones.include_env(xml_string)

### Load robot
robot = QuadrupedWrapperAbstract(
    URDF_path,
    xml_string,
    package_dir,
    rotor_inertia=cfg.rotor_inertia,
    gear_ratio=cfg.gear_ratio,
    foot_size=cfg.foot_size,
)

### Simulator
simulator = SteppingStonesSimulator(stepping_stones, robot, None, height_offset = 0.01)

### Set jump start and goal
id_contacts_plan = np.array([
    [26, 6, 24, 4],
    [26, 6, 24, 4],
    [27, 7, 24, 4],
    [27, 7, 25, 5],
    [28, 8, 25, 5],
    [28, 8, 26, 6],
    ])

simulator.set_start_and_goal(start_indices=id_contacts_plan[0], goal_indices=[])

### Load regressor model
regressor = load_model(REGRESSOR_PATH)
classifier = load_model(CLASSIFIER_PATH)
q, v = robot.get_pin_state()

offset_pos_w = np.zeros((4, 3))

with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
    
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    
    viewer.sync()
    sim_start_time = time.time()
    while (viewer.is_running()):
        
        for i in range(len(id_contacts_plan) - 1):
            start_pos_w = stepping_stones.positions[id_contacts_plan[i]]
            target_pos_w = stepping_stones.positions[id_contacts_plan[i+1]]
            position_3d_callback(viewer, target_pos_w + offset_pos_w)
            
            success, proba_success = is_feasible(classifier, q, v, start_pos_w, target_pos_w, 0.7)
            
            robot.step()
            viewer.sync()
            time.sleep(2)
        
            q, v, offset_pos_w = predict_next_state(regressor, q, v, robot, start_pos_w, target_pos_w)

            robot.mj_data.qpos = robot.pin2mj_state(copy.deepcopy(q))
            robot.mj_data.qvel = v