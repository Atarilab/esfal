import os
# USe GPU rendering
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import argparse

from mpc_controller.bicon_mpc_offset import BiconMPCOffset
from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.simulator import Simulator

from environment.stepping_stones import SteppingStonesEnv
from environment.sim import SteppingStonesSimulator

from utils.visuals import desired_contact_locations_callback
from tree_search.mcts_stepping_stones import MCTSSteppingStonesKin, MCTSSteppingStonesDyn

REGRESSOR_PATH = f"/home/atari_ws/project/tree_search/trained_models/state_estimator/1/MLP.pth"
CLASSIFIER_PATH = f"/home/atari_ws/project/tree_search/trained_models/classifier/0/MLP.pth"


class Go2Config:
    name = "go2"
    mesh_dir = "assets"
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    gear_ratio = 6.33
    foot_size = 0.02

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argument parser for simulation parameters.")
    parser.add_argument('--mode', type=str, default="dyn", help='Which MCTS to use (kin or dyn)')
    args = parser.parse_args()
    
    cfg = Go2Config

    ### Paths
    URDF_path,\
    xml_string,\
    package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir)
    
    ### Stepping stones env
    stepping_stones_height = 0.05
    stepping_stones = SteppingStonesEnv(
        grid_size=(7, 4),
        spacing=(0.18, 0.28/2),
        size_ratio=(0.8, 0.8),
        height=stepping_stones_height,
        randomize_pos_ratio=0.,
        randomize_size_ratio=[0.55, 0.55],
        shape="cylinder"
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
    
    ### Controller
    controller = BiconMPCOffset(robot, REGRESSOR_PATH, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
    controller.set_gait_params(jump)  # Choose between trot, jump and bound

    ### Simulator
    simulator = SteppingStonesSimulator(stepping_stones, robot, controller)

    ### MCTS
    start = [23, 9, 21, 7]
    goal = [25, 11, 23, 9]
    
    if args.mode == "kin":
        mcts = MCTSSteppingStonesKin(
            simulator,
            simulation_steps=2,
            alpha_exploration=0.,
            C=5.,
            W=1.,
            max_solution_search=2,
            print_info=True
        )
    elif args.mode == "dyn":
        ### MCTS

        mcts = MCTSSteppingStonesDyn(
            simulator,
            simulation_steps=1,
            alpha_exploration=0.0,
            C=1.,
            W=1.,
            state_estimator_state_path=REGRESSOR_PATH,
            classifier_state_path=CLASSIFIER_PATH,
            max_solution_search=2,
            feaibility=.2,
            accuracy=.01,
            print_info=True,
        )
        # Important to init the robot position
        simulator.set_start_and_goal(start_indices=start, goal_indices=goal)

    mcts.search(start, goal, num_iterations=10000)
    
    for fn_name, timings in mcts.get_timings().items():
        print(fn_name, timings)
    
    print(mcts.solutions)
    
    visual_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller)
        )
    
    simulator.run_contact_plan(
        mcts.solutions[0], 
        use_viewer=False, 
        visual_callback_fn=visual_callback,
        
        record_video=True,
        fps=30,
        video_path="test.mp4",
        playback_speed=0.5,
        frame_height=1080, frame_width=1920)