import os
# USe GPU rendering
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import argparse

from mpc_controller.bicon_mpc_offset import BiconMPCOffset
from mpc_controller.bicon_mpc import BiConMPC
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
from learning_jump_feasibility.collect_data import RecordJumpData

# REGRESSOR_PATH = f"/home/atari_ws/project/tree_search/trained_models/state_estimator/1/MLP.pth"
# CLASSIFIER_PATH = f"/home/atari_ws/project/tree_search/trained_models/classifier/0/MLP.pth"
REGRESSOR_PATH = "learning_jump_feasibility/logs/MLP_regressor/0/MLP.pth"
CLASSIFIER_PATH = "learning_jump_feasibility/logs/MLPclassifierBinary/0/MLP.pth"
OFFSET_PATH = "learning_jump_feasibility/logs/MLP_offset/1/MLP.pth"


class Go2Config:
    name = "go2"
    mesh_dir = "assets"
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    gear_ratio = 6.33
    foot_size = 0.02

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argument parser for simulation parameters.")
    parser.add_argument('--mode', type=str, default="dyn", help='Which MCTS to use (kin or dyn)')
    parser.add_argument('--offset', type=bool, default=True, help='Use offset controller')
    args = parser.parse_args()
    
    cfg = Go2Config

    ### Paths
    URDF_path,\
    xml_string,\
    package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir)
    
    ### Stepping stones env
    stepping_stones_height = 0.2
    stepping_stones = SteppingStonesEnv(
        grid_size=(7, 5),
        spacing=(0.18, 0.28/2),
        size_ratio=(0.50, 0.50),
        height=stepping_stones_height,
        shape="cylinder"
    )

    start = [23, 9, 21, 7]
    goal = [27, 13, 25, 11]
   
   # get current random state of numpy
    state = np.random.get_state()
    np.random.seed(1)
    # randomize stones position
    stepping_stones.remove_random(N_to_remove=9, keep=[start, goal])
    stepping_stones.randomize_center_location(0.75, keep=[start, goal])
    print(stepping_stones.id_to_remove)
    # set the random state back to the original
    np.random.set_state(state)
    
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
    controller = BiconMPCOffset(robot, OFFSET_PATH, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
    # controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
    controller.set_gait_params(jump)  # Choose between trot, jump and bound

    # data recorder
    data_recorder = RecordJumpData(robot, None)

    ### Simulator
    simulator = SteppingStonesSimulator(stepping_stones, robot, controller, data_recorder, update_data_recorder=True)

    if args.mode == "kin":
        mcts = MCTSSteppingStonesKin(
            simulator,
            simulation_steps=2,
            alpha_exploration=0.0,
            C=0.01,
            W=5.,
            max_solution_search=3,
            print_info=True,
            n_threads_kin=1,
            n_threads_sim=1,
            use_inverse_kinematics=False,
            state_estimator_state_path=REGRESSOR_PATH,
            classifier_state_path=CLASSIFIER_PATH,
            # simulation='network',
            # simulation='mpc',
            # network_simulation_threshold=0.65
        )
        # Important to init the robot position
        simulator.set_start_and_goal(start_indices=start, goal_indices=goal)

    elif args.mode == "dyn":
        ### MCTS

        mcts = MCTSSteppingStonesDyn(
            simulator,
            simulation_steps=2,
            alpha_exploration=0.0,
            C=0.01,
            W=5.,
            state_estimator_state_path=REGRESSOR_PATH,
            classifier_state_path=CLASSIFIER_PATH,
            max_solution_search=3,
            classifier_threshold=0.6,
            safety=0.8,
            accuracy=0.2,
            print_info=True,
        )
        # Important to init the robot position
        simulator.set_start_and_goal(start_indices=start, goal_indices=goal)

    mcts.search(start, goal, num_iterations=10000)
    
    for fn_name, timings in mcts.get_timings().items():
        print(fn_name, timings)
    
    print(mcts.statistics)
    # print(mcts.solutions)
    
    # test_solutions = []
    # for solution in mcts.solutions:
    #     goal_reached = simulator.run_contact_plan(
    #         solution, 
    #         use_viewer=False, 
    #         record_video=False)
        
    #     test_solutions.append(goal_reached)
    
    # print(test_solutions)

    # print()
    # print(stepping_stones.positions[mcts.solutions[2]])
    # print(data_recorder.record_feet_contact)

    simulator.run_contact_plan(
        mcts.solutions[0], 
        use_viewer=False, 
        
        record_video=True,
        fps=30,
        video_path="test.mp4",
        playback_speed=0.5,
        frame_height=1080, frame_width=1920)