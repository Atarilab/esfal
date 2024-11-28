import numpy as np
import time
from mujoco._structs import MjData

from mpc_controller.cyclic_gait_gen import CyclicQuadrupedGaitGen
from mpc_controller.robot_id_controller import InverseDynamicsController
from mpc_controller.motions.weight_abstract import BiconvexMotionParams
from mpc_controller.bicon_mpc import BiConMPC
from mj_pin_wrapper.abstract.robot import RobotWrapperAbstract
from mj_pin_wrapper.abstract.controller import ControllerAbstract

from learning_jump_feasibility.test.test_utils import load_model, compute_offsets

class BiconMPCOffset(BiConMPC):
    MIN_STEP_IN_CONTACT = 2
    
    def __init__(self, robot: RobotWrapperAbstract, state_predictor_path : str, **kwargs) -> None:
        super().__init__(robot, **kwargs)
        self.state_predictor = load_model(state_predictor_path)
        self.state_predictor.eval()
        self.consecutive_landing = 0
        self.waiting_for_next_jump = True

        self.half_gait = False

    def reset(self):
        self.consecutive_landing = 0
        self.waiting_for_next_jump = True

        self.half_gait = False

        return super().reset()
    
    def _update_consecutive_landing(self) -> bool:
        eeff_contact = self.robot.get_mj_eeff_contact_with_floor()
        N_contact = sum(eeff_contact.values())
        landed = N_contact == 4
        # Robot jumping or in the air
        if not landed:
            self.consecutive_landing = 0
            self.waiting_for_next_jump = True
        # Robot making contact or staying in contact
        else:
            self.consecutive_landing += 1
        # print(self.consecutive_landing)
        return landed
    
    def get_desired_contacts(self, q, v) -> np.ndarray:
        mpc_contacts = []
        if len(self.contact_plan_des) > 0:
            
            # Stay on the last contact location if end of contact plan is reached
            if self.replanning + 2 * self.gait_horizon > len(self.full_length_contact_plan):
                self.full_length_contact_plan = np.concatenate(
                        (
                        self.full_length_contact_plan,
                        np.repeat(self.full_length_contact_plan[-1, np.newaxis, :, :], 2 * self.gait_horizon,
                        axis=0
                        )),
                    axis=0
                )

                self.achieved = True

            # Take the next <horizon> contact locations
            mpc_contacts = self.full_length_contact_plan[self.replanning: self.replanning + self.gait_horizon]
            # Update the desired velocity
            i_next_jump = self.replanning + 2 * (self.gait_horizon - 2)
            center_position_next_cnt = np.mean(self.full_length_contact_plan[i_next_jump], axis=0)
            self.v_des = np.round((center_position_next_cnt - q[:3]) / self.gait_period, 2)
            # Scale velocity
            # self.v_des *= np.array([1.3, 2., 0.])
            self.v_des *= np.array([1.3, 2.0, 0.])
            # print("Desired velocity: ", self.v_des)
            
            self._update_consecutive_landing()
            if (self.consecutive_landing > BiconMPCOffset.MIN_STEP_IN_CONTACT and
                self.waiting_for_next_jump):

                # if self.half_gait:
                #     self.half_gait = False
                #     self.waiting_for_next_jump = False
                    
                #     self.replanning += 1
                #     return mpc_contacts
                
                self.waiting_for_next_jump = False
                self.half_gait = True
                
                # Run the offset network and update the contact locations
                i_target = self.gait_horizon - self.replanning % self.gait_horizon 
                start_pos_w = mpc_contacts[0]
                target_pos_w = mpc_contacts[i_target]
                mpc_offset_w = compute_offsets(self.state_predictor, q, v, start_pos_w, target_pos_w)

                # Apply offset on the next target position
                mpc_contacts[i_target: i_target + self.gait_horizon, :, :] += mpc_offset_w

        self.replanning += 1
        return mpc_contacts