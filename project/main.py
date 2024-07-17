import os
# USe GPU rendering
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np

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



class Go2Config:
    name = "go2"
    mesh_dir = "assets"
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    gear_ratio = 6.33
    foot_size = 0.02

if __name__ == "__main__":
    
    cfg = Go2Config

    ### Paths
    URDF_path,\
    xml_string,\
    package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir)
    
    ### Stepping stones env
    stepping_stones_height = 0.2
    start = [23, 9, 21, 7]
    goal = [27, 13, 25, 11]

    stepping_stones = SteppingStonesEnv(
        grid_size=(7, 5),
        spacing=(0.18, 0.28/2),
        size_ratio=(0.5, 0.5),
        height=stepping_stones_height,
        shape="cylinder",
        start=start,
        goal=goal,
    )

    state = np.random.get_state()
    np.random.seed(2)
    # randomize stones position

    stepping_stones.remove_random(N_to_remove=9, keep=[start, goal])
    stepping_stones.randomize_center_location(0.75, keep=[start, goal])
    print(stepping_stones.id_to_remove)
    # set the random state back to the original
    np.random.set_state(state)
    
    id_contacts_plan = np.array([
        [23, 9, 21, 7], [24, 3, 29, 1], [25, 4, 23, 2], [33, 12, 24, 3], [27, 13, 25, 11]
    ])

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
    # MODEL_PATH = "learning_jump_feasibility/logs/MLP_regressor/1/MLP.pth"
    MODEL_PATH = "learning_jump_feasibility/logs/MLP_offset/1/MLP.pth"
    # controller = BiconMPCOffset(robot, MODEL_PATH, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
    controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
    controller.set_gait_params(jump)  # Choose between trot, jump and bound

    ### Simulator
    simulator = SteppingStonesSimulator(stepping_stones, robot, controller)
    simulator.set_start_and_goal(start_indices=start, goal_indices=goal)

    visual_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller)
        )

    # Run
    goal_reached = simulator.run_contact_plan(
        id_contacts_plan,
        use_viewer=False,
        visual_callback_fn=visual_callback,
        
        record_video=True,
        fps=30,
        video_save_path="test.mp4",
        playback_speed=0.5,
        frame_height=1080, frame_width=1920,
        )
    
    if goal_reached: print("Goal reached.")
    else: print("Failed")
