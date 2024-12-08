import os
# USe GPU rendering
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import argparse

from mpc_controller.jump.bicon_mpc_offset import BiconMPCOffset as BiconMPCOffset_jump
from mpc_controller.jump.bicon_mpc import BiConMPC as BiConMPC_jump
from mpc_controller.trot.bicon_mpc_offset import BiconMPCOffset as BiconMPCOffset_trot
from mpc_controller.trot.bicon_mpc import BiConMPC as BiConMPC_trot
from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.simulator import Simulator

from environment.stepping_stones import SteppingStonesEnv
from environment.sim import SteppingStonesSimulator

from utils.visuals import desired_contact_locations_callback
from tree_search.mcts_stepping_stones import MCTSSteppingStonesKin, MCTSSteppingStonesDyn
from learning_jump_feasibility.collect_data import RecordJumpData



class Go2Config:
    name = "go2"
    mesh_dir = "assets"
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    gear_ratio = 6.33
    foot_size = 0.02

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argument parser for simulation parameters.")
    parser.add_argument('--gait', type=str, default="jump", help='Which MCTS to use (kin or dyn)')
    parser.add_argument('--mode', type=str, default="dyn", help='Which MCTS to use (kin or dyn)')
    parser.add_argument('--offset', type=bool, default=True, help='Use offset controller')
    parser.add_argument('--size', type=int, default=50, help='Size of the stones')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    
    REGRESSOR_PATH = f"tree_search/trained_models/{args.gait}/state_estimator/0/MLP.pth"
    CLASSIFIER_PATH = f"tree_search/trained_models/{args.gait}/classifier/0/MLP.pth"
    OFFSET_PATH = f"tree_search/trained_models/{args.gait}/offset/0/MLP.pth"
    
    cfg = Go2Config

    ### Paths
    URDF_path,\
    xml_string,\
    package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir)

    start = [23, 9, 21, 7]
    goal = [27, 13, 25, 11]

    size = args.size / 100.0
    
    ### Stepping stones env
    stepping_stones_height = 0.2
    stepping_stones = SteppingStonesEnv(
        grid_size=(7, 5),
        spacing=(0.18, 0.28/2),
        size_ratio=(size, size),
        height=stepping_stones_height,
        shape="box",
        start=start,
        goal=goal,
    )
   
   # get current random state of numpy
    state = np.random.get_state()
    np.random.seed(args.seed)
    # randomize stones position
    stepping_stones.remove_random(N_to_remove=9, keep=[start, goal])
    stepping_stones.randomize_center_location(0.75, keep=[start, goal])
    stepping_stones.randomize_height(0.02, keep=[start, goal])
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

    if args.gait == "trot":
        if args.offset:
            controller = BiconMPCOffset_trot(robot, OFFSET_PATH, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
        else:
            controller = BiConMPC_trot(robot, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
        controller.set_gait_params(trot)  # Choose between trot, jump and bound
    elif args.gait == "jump":
        if args.offset:
            controller = BiconMPCOffset_jump(robot, OFFSET_PATH, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
        else:
            controller = BiConMPC_jump(robot, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
        controller.set_gait_params(jump)

    # data recorder
    data_recorder = RecordJumpData(robot, None)

    ### Simulator
    check_goal_period = 750 if args.gait == "trot" else 500
    simulator = SteppingStonesSimulator(stepping_stones, robot, controller, data_recorder, update_data_recorder=True, check_goal_period=check_goal_period)

    if args.mode == "kin":
        mcts = MCTSSteppingStonesKin(
            simulator,
            simulation_steps=2,
            alpha_exploration=0.0,
            C=0.01,
            W=5.,
            max_solution_search=1,
            print_info=True,
            n_threads_kin=1,
            n_threads_sim=1,
            use_inverse_kinematics=False,
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
            max_solution_search=1,
            classifier_threshold=0.65,
            safety=0.0,
            accuracy=0.0,
            print_info=True,
        )
        # Important to init the robot position
        simulator.set_start_and_goal(start_indices=start, goal_indices=goal)

    mcts.search(start, goal, num_iterations=20000)
    
    for fn_name, timings in mcts.get_timings().items():
        print(fn_name, timings)
    
    print(mcts.statistics)
    print(mcts.solutions)

    simulator.run_contact_plan(
        mcts.solutions[0], 
        use_viewer=False, 
        
        record_video=True,
        fps=30,
        video_path="test.mp4",
        playback_speed=0.5,
        frame_height=1080, frame_width=1920)