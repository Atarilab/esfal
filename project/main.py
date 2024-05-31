import numpy as np

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
    stepping_stones_height = 0.1
    stepping_stones = SteppingStonesEnv(
        grid_size=(10, 3),
        spacing=(0.18, 0.14),
        size_ratio=(0.75, 0.85),
        height=stepping_stones_height,
        randomize_pos_ratio=0.,
        randomize_size_ratio=[0.5, 0.6]
    )

    id_contacts_plan = np.array([
        [26, 6, 24, 4],
        [26, 6, 24, 4],
        [27, 7, 24, 4],
        [27, 7, 24, 4],
        [27, 7, 25, 5],
        [28, 8, 25, 5],
        [28, 8, 25, 5],
        [28, 8, 26, 6],
        [28, 8, 26, 6],
        [28, 8, 26, 6],
        [28, 8, 26, 6],
        [28, 8, 26, 6],
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
    controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False, height_offset=stepping_stones_height)
    controller.set_gait_params(jump)  # Choose between trot, jump and bound

    ### Simulator
    simulator = SteppingStonesSimulator(stepping_stones, robot, controller)

    visual_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller)
        )

    # Run
    goal_reached = simulator.run_contact_plan(
        id_contacts_plan,
        use_viewer=False,
        visual_callback_fn=visual_callback
        )
    
    if goal_reached: print("Goal reached.")
    else: print("Failed")
