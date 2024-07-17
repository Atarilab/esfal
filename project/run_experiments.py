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


REGRESSOR_PATH = f"tree_search/trained_models/state_estimator/0/MLP.pth"
CLASSIFIER_PATH = f"tree_search/trained_models/classifier/0/MLP.pth"
OFFSET_PATH = f"tree_search/trained_models/offset/1/MLP.pth"
# REGRESSOR_PATH = "learning_jump_feasibility/logs/MLP_regressor/0/MLP.pth"
# CLASSIFIER_PATH = "learning_jump_feasibility/logs/MLPclassifierBinary/0/MLP.pth"
# OFFSET_PATH = "learning_jump_feasibility/logs/MLP_offset/1/MLP.pth"

class Go2Config:
    name = "go2"
    mesh_dir = "assets"
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    gear_ratio = 6.33
    foot_size = 0.02

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argument parser for simulation parameters.")
    parser.add_argument('--mode', type=str, default="dyn", help='Which MCTS to use (kin or dyn)')
    parser.add_argument('--num_remove', type=int, default=9, help='Number of stones to remove')
    parser.add_argument("--pos_noise", type=float, default=0.0, help="Stepping stones position noise")
    parser.add_argument('--size_ratio', type=float, default=0.7, help='Size ratio of stones')
    parser.add_argument('--offset', type=bool, default=False, help='Use offset controller')
    parser.add_argument('--alpha', type=float, default=0.0, help='Alpha safety value')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta accuracy value')
    parser.add_argument('--id', type=int, default=0, help='Episode id')
    args = parser.parse_args()

    print(args)
    
    cfg = Go2Config
    
    num_episodes = 140
    num_repeat = 1

    save_dir = f"/home/atari_ws/results/runs/run_{args.mode}_{'offset' if args.offset else ''}_{int(args.size_ratio*100)}_{int(args.alpha*100)}_{int(args.beta*100)}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Count: {args.id}")
    if not os.path.exists(f"{save_dir}/{args.id}.npz"):
        id = args.id
        print(f"Episode: {id}")
        for repeat in range(num_repeat):
            
            ### Paths
            URDF_path,\
            xml_string,\
            package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir)

            ### Stepping stones env
            stepping_stones_height = 0.2
            stepping_stones = SteppingStonesEnv(
                grid_size=(7, 5),
                spacing=(0.18, 0.28/2),
                size_ratio=(args.size_ratio, args.size_ratio),
                height=stepping_stones_height,
                shape="cylinder"
            )
            start = [23, 9, 21, 7]
            goal = [27, 13, 25, 11]
        
        # get current random state of numpy
            state = np.random.get_state()
            np.random.seed(id)
        
            stepping_stones.remove_random(N_to_remove=args.num_remove, keep=[start, goal])
            # randomize stones position
            stepping_stones.randomize_center_location(args.pos_noise, keep=[start, goal])

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
            if args.offset:
                controller = BiconMPCOffset(robot, OFFSET_PATH, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
            else:
                controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)

            controller.set_gait_params(jump)  # Choose between trot, jump and bound

            # data recorder
            data_recorder = RecordJumpData(robot, None)

            ### Simulator
            simulator = SteppingStonesSimulator(stepping_stones, robot, controller, data_recorder, update_data_recorder=True)

            ### MCTS
            
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
                )
                # Important to init the robot position
                simulator.set_start_and_goal(start_indices=start, goal_indices=goal)
            elif args.mode == "dyn":

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
                    safety=args.alpha,
                    accuracy=args.beta,
                    print_info=True,
                )
                # Important to init the robot position
                simulator.set_start_and_goal(start_indices=start, goal_indices=goal)

            mcts.search(start, goal, num_iterations=10000)
                    
            print(mcts.statistics)

            np.savez(f"{save_dir}/{args.id}.npz", {
                'statistics': mcts.statistics,
                'solutions': mcts.solutions,
                'id': id,
                'repeat': repeat,
                'mode': args.mode,
                'num_remove': args.num_remove,
                'pos_noise': args.pos_noise,
                'offset': args.offset,
                'id_remove': stepping_stones.id_to_remove,
            })