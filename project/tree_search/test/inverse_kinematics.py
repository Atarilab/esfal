import time
from typing import Tuple
import pinocchio
import mujoco
import numpy as np
import mujoco
from copy import deepcopy
from utils.visuals import position_3d_callback
from numpy.linalg import norm
from scipy.linalg import solve 

from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from robots.configs import Go2Config

EPS    = 3e-2
IT_MAX = 150
DT     = 0.5
DT_INCR   = 1. / IT_MAX
DAMP   = 1e-14
CLIP_LIN_VELOCITY = 0.045
CLIP_ANG_VELOCITY = 0.35
CLIP_J_VELOCITY = 0.4
SCALE_RAND_POS = 0.1
CLIP_ARRAY = np.array(
    [CLIP_LIN_VELOCITY] * 3 +
    [CLIP_ANG_VELOCITY] * 3 +
    [CLIP_J_VELOCITY] * 12
)

def ik_iteration(model, data, q: np.ndarray, frame_ids_feet, oMdes_feet, dt) -> Tuple[bool, np.ndarray]:
    """
    Perform one inverse kinematics iteration.
    Return:
        - bool: success
        - np.ndarray: configuration
    """
    err = np.zeros(6 * len(frame_ids_feet))
    J_full = np.zeros((6 * len(frame_ids_feet), model.nv))
    
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.updateFramePlacements(model, data)

    for k, (frame_id, oMdes) in enumerate(zip(frame_ids_feet, oMdes_feet)):
        oMcurrent = data.oMf[frame_id]
        dMi = oMdes.actInv(oMcurrent)
        err[6*k:6*k+3] = pinocchio.log(dMi).vector[:3] # Just position error
        J_full[6*k:6*(k+1), :] = pinocchio.computeFrameJacobian(model, data, q, frame_id)
    
    if norm(err) < EPS:
        return True, q

    v = - J_full.T.dot(solve(J_full.dot(J_full.T) + DAMP * np.eye(J_full.shape[0]), err))
    
    v = np.clip(v, -CLIP_ARRAY, CLIP_ARRAY)
    
    q = pinocchio.integrate(model, q, v * dt)
        
    return False, q

def ik_solver(model, data, frame_ids_feet, oMdes_feet, q0):
    
    success = False
    i = 0
    q = q0.copy()
    dt = DT
    
    while True:
        if i >= IT_MAX:
            success = False
            break
        
        success, q = ik_iteration(model, data, q, frame_ids_feet, oMdes_feet, dt)
        dt += DT_INCR
        if success:
            break
        
        i += 1
        
    return success, q

def compute_timings(model, data, N_runs:int = 2000):
    
    feet_pin_frame_name = robot.pin_feet_frame_name
    frame_ids_feet = [model.getFrameId(frame_name) for frame_name in feet_pin_frame_name]
    feet_position_world = robot.get_pin_feet_position_world()
    
    # Init
    q0, _   = robot.get_pin_state()
    q0[6:] /= 2.
    q0[2] += .1
    
    # Runs
    run_times = []
    run_times_success = []
    successes = []
    for _ in range(N_runs):
        
        ### Create randomized desired feet positions
        desired_feet_position = [
            pos + np.random.randn(3) * SCALE_RAND_POS for pos in feet_position_world
            ]
        
        oMdes_feet = [
            pinocchio.SE3(np.eye(3), desired_pos)
            for desired_pos in desired_feet_position
            ]
    
        t = time.time()
        success, _ = ik_solver(model, data, frame_ids_feet, oMdes_feet, q0)
        run_time = time.time() - t
        
        run_times.append(run_time)
        if success:
            run_times_success.append(run_time)
        successes.append(int(success))
    
    print("Mean solving time", sum(run_times) / N_runs * 1000, "ms")
    print("Mean solving time success", sum(run_times_success) / sum(successes) * 1000, "ms")
    print("Success", sum(successes) / N_runs * 100, "%" )

def visualize_ik(model, data):
    
    ### Create randomized desired feet positions
    feet_pin_frame_name = robot.pin_feet_frame_name
    frame_ids_feet = [model.getFrameId(frame_name) for frame_name in feet_pin_frame_name]
    feet_position_world = robot.get_pin_feet_position_world()

    desired_feet_position = [
        pos + np.random.randn(3) / 30. for pos in feet_position_world
        ]
    
    oMdes_feet = [
        pinocchio.SE3(np.eye(3), desired_pos)
        for desired_pos in desired_feet_position
        ]
    
    # Init
    q, _   = robot.get_pin_state()
    q[6:] /= 2.
    q[2] += .1
    i = 0
    dt = DT
    success = False
    
    # With viewer
    mj_q = q.copy()
    mj_q[3:7] = q[6], q[3], q[4], q[5]
    robot.reset(mj_q)
    
    with mujoco.viewer.launch_passive(robot.mj_model, robot.mj_data) as viewer:
        
        # Enable wireframe rendering of the entire scene.
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        
        viewer.sync()

        while (viewer.is_running()):
                        
            if not success:
                success, q = ik_iteration(model, data, q, frame_ids_feet, oMdes_feet, dt)
                dt += DT_INCR
                mj_q = deepcopy(q)
                mj_q[3:7] = q[6], q[3], q[4], q[5]
                robot.mj_data.qpos = mj_q
                
            if success:
                print("Success", i)
                break
                
            else:
                i += 1
                viewer.sync()
                
            robot.step()
            position_3d_callback(viewer, desired_feet_position)
            time.sleep(0.1)


if __name__ == "__main__":
    
    cfg = Go2Config

    ### Paths
    URDF_path,\
    xml_string,\
    package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir, models_path="./robots")

    ### Load robot
    robot = QuadrupedWrapperAbstract(
        URDF_path,
        xml_string,
        package_dir,
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        foot_size=cfg.foot_size,
        )
    robot.reset()

    model  = robot.pin_model
    data  = robot.pin_data
    
    compute_timings(model, data)
    visualize_ik(model, data)