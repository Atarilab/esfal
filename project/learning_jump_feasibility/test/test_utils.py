import os
import glob
import torch
import pinocchio as pin
import copy
import numpy as np

from learning_jump_feasibility.data.dataset_supervised import transform_points
from learning_jump_feasibility.models.MLP import MLP
from learning_jump_feasibility.utils.config import Config


def load_model(state_path: str):
    """
    Load model from state path.
    """
    run_dir = os.path.split(state_path)[0]
    config_path = glob.glob(run_dir + "/*.yaml") + glob.glob(run_dir + "/*.yml")
    assert len(config_path) > 0, f"Config file not found in {run_dir}"
    cfg = Config(config_path[0])

    cfg_model = cfg.model["PARAMS"]

    model = MLP(**cfg_model)
    state = torch.load(state_path, map_location=torch.device('cpu'))
    model.load_state_dict(state["state_dict"])
    
    return model


def align_3d_points(A, B):
    """
    Compute transform to minimize distance
    between 2 sets of points.
    """

    # https://stackoverflow.com/questions/60877274/optimal-rotation-in-3d-with-kabsch-algorithm
    t = np.mean(A, axis=0) - np.mean(B, axis=0)
    h = A.T @ B

    u, s, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    r = v @ e @ u.T
    
    return t, r
    
def predict_next_state(model, q, v, robot, start_pos_w, target_pos_w):
    
    ### Predict next state with the network
    W_T_b = pin.XYZQUATToSE3(q[:7])
    b_T_W = W_T_b.inverse()
    feet_pos_b = transform_points(b_T_W, start_pos_w).reshape(-1)
    target_pos_b = transform_points(b_T_W, target_pos_w).reshape(-1)
    state = np.concatenate((q[3:], v[:6]), axis=0) # Should have the right shape depending on the model input size
    input = torch.from_numpy(np.concatenate((state, target_pos_b, feet_pos_b), axis=0)).float().unsqueeze(0)

    with torch.no_grad():
        out = model(input).squeeze()
        
    q_pred, v_pred, mpc_input_pos_b = torch.split(out, [16, 6, 12], dim=-1)

    ### Find transform for base pose to align feet location in current configuration
    ### and target locations

    # Set robot to predicted configuration
    robot_copy = copy.copy(robot)

    q_pred = np.concatenate((q[:3], q_pred.tolist()))
    q_pred[3:7] = np.abs(q_pred[3:7])
    q_pred[3:7] /= np.linalg.norm(q_pred[3:7])
    q_pred_mj = robot.pin2mj_state(q_pred)

    robot_copy.reset(q_pred_mj)
    
    # Feet position in the predicted configuration
    feet_pos_pred_w = robot_copy.get_pin_feet_position_world()

    # Compute transform to align feet and target positions
    t, r = align_3d_points(target_pos_w, feet_pos_pred_w)

    # Apply transformation from current base pose
    b0_T_b1 = pin.XYZQUATToSE3(np.zeros(7))
    b0_T_b1.translation = t
    b0_T_b1.rotation = np.eye(3)

    W_T_b1 = W_T_b * b0_T_b1

    q_pred[:7] = pin.SE3ToXYZQUAT(W_T_b1)
    
    # Express mpc input in world frame knowing the current pose
    mpc_input_pos_w = transform_points(W_T_b1, mpc_input_pos_b.reshape(4,3))
    
    # Append joint vel
    v_pred = np.concatenate((v_pred, np.zeros(12)))
    
    return q_pred, v_pred, mpc_input_pos_w

    
def is_feasible(classifier, q, v, start_pos_w, target_pos_w, threshold) -> np.ndarray:
        
    ### Compute classification score with the network
    W_T_b = pin.XYZQUATToSE3(q[:7])
    b_T_W = W_T_b.inverse()
    feet_pos_b = transform_points(b_T_W, start_pos_w.reshape(-1, 3)).reshape(-1, 12)
    target_pos_b = transform_points(b_T_W, target_pos_w.reshape(-1, 3)).reshape(-1, 12)
    B = len(target_pos_b)
    
    state = np.concatenate((q[3:], v[:6]), axis=0) # Should have the right shape depending on the model input size
    state_batched = np.repeat((state[np.newaxis, :]), B, axis=0) # Batch the input
    
    input = torch.from_numpy(np.concatenate((state_batched, target_pos_b, feet_pos_b), axis=-1)).float()

    with torch.no_grad():
        score = classifier(input).squeeze()
        
    proba_success = torch.nn.functional.sigmoid(score).numpy()
    success = proba_success > threshold
    
    return success, proba_success
        