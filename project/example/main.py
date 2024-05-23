import numpy as np
import mujoco
import pinocchio

from motions.cyclic.go2_trot import trot
from motions.cyclic.go2_jump import jump
from motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.simulator import Simulator
from environment.stepping_stones import SteppingStonesEnv
from utils.visuals import desired_contact_locations_callback

class Go2Config:
    name = "go2"
    mesh_dir = "assets"
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    gear_ratio = 6.33
    foot_size = 0.01

if __name__ == "__main__":
    
    cfg = Go2Config

    ### Paths
    URDF_path,\
    xml_string,\
    package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir)
    
    ### Stepping stones env
    stepping_stones = SteppingStonesEnv(
        grid_size=(9, 9),
        spacing=(0.21, 0.14),
        size_ratio=(0.35, 0.55),
        height=0.01,
        randomize_pos_ratio=0.5,
        randomize_size_ratio=[0.4, 0.6]
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
    q0, _ = robot.get_mj_state()
    q0[2] += stepping_stones.height + 0.02
    robot.reset(q0)

    ### Controller
    controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False)
    
    v_des = np.array([0.0, 0.0, 0.0])
    w_des = 0.0

    controller.set_command(v_des, w_des)
    controller.set_gait_params(trot)  # Choose between trot, jump and bound

    ### Simulator
    simulator = Simulator(robot, controller)
    
    visual_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller)
        )

    # Run 
    SIM_TIME = 10 #s
    simulator.run(
        simulation_time=SIM_TIME,
        visual_callback_fn=visual_callback,
        use_viewer=True,
        real_time=False,
        verbose=False
    )