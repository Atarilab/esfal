import copy
import time
import pinocchio as pin
from multiprocessing import Manager, Pool, Queue
import argparse
import numpy as np
import os
from tqdm import tqdm
from mujoco._structs import MjData

from numpy.core.multiarray import array as array

from mpc_controller.motions.cyclic.go2_jump import jump
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.bicon_mpc import BiConMPC
from environment.stepping_stones import SteppingStonesEnv
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
from utils.visuals import desired_contact_locations_callback, position_3d_callback
from robots.configs import Go2Config
from tree_search.kinematics import QuadrupedKinematicFeasibility

# DEFAULT_PATH = "/home/atari_ws/data/learning_jump_feasibility"
DEFAULT_PATH = "/home/akizhanov/esfal/data/learning_jump_feasibility"
V = .08
SCALE_NOISE = 0.12
# SCALE_NOISE = 0.01
CHANGE_DIR_STEP = 2
LENGTH = 100

### Data recorder
class RecordJumpData(DataRecorderAbstract):
    MIN_STEP_IN_CONTACT = 175 # ms
    FILE_NAME = "data.npz"
    STATE_NAME = "state"
    CONTACT_NAME = "contact"
    TARGET_NAME = "target"
    COLLISION_NAME = "collision"
    
    def __init__(self, robot : QuadrupedWrapperAbstract, contact_plan, record_dir: str = "") -> None:
        super().__init__(record_dir)
        self.robot = robot
        self.contact_plan = contact_plan
        self.consecutive_landing = 0
        self.waiting_for_next_jump = True
        self.i_jump = 0
        
        self.saving_file_path = os.path.join(record_dir, RecordJumpData.FILE_NAME)
    
        # [x, y, z, qx, qy, qz, qw, v, w, qj, v_j]
        self.record_state = []
        # [c1, c2, c3, c4] in world frame
        self.record_feet_contact = []
        # [c1, c2, c3, c4] stepping stones in world frame
        self.record_target_contact = []
        # [knee_FL, knee_FR, knee_RL, knee_RR, other] 1 if collision, 0 otherwise
        self.record_collision = []
                    
    def reset(self) -> None:
        self.consecutive_landing = 0
        self.waiting_for_next_jump = True
        self.i_jump = 0
        self.record_state = []
        self.record_feet_contact = []
        self.record_target_contact = []
        self.record_collision = []

    def update_contact_plan(self, contact_plan) -> None:
        self.contact_plan = contact_plan
        
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
    
    def _get_collision_id(self, cnt_pos_b : np.array) -> int:
        """
        Returns id of the contact array from the contact position in base frame.
        0: FL_knee, 1: FL_knee, 1: FL_knee, 3: FL_knee,
        """
        x, y = cnt_pos_b[0], cnt_pos_b[1]
        id = None
        if x >= 0 and y >= 0:
            id = 0
        elif x >= 0 and y <= 0:
            id = 1
        elif x <= 0 and y >= 0:
            id = 2
        elif x <= 0 and y <= 0:
            id = 3
        return id
    
    def _check_collision(self) -> np.array:
        """
        Return an array with bool indicating if parts of the robots are in collision
        with statics objects.
        0: no conctact
        1: contact

        Returns:
            np.array: [FL_knee, FR_knee, RL_knee, RR_knee, others]
        """
        collision_contacts = np.zeros(5, dtype=np.int8)
        if self.robot.is_collision():
            # Filter contacts
            for cnt in self.robot.mj_data.contact:
                if ((cnt.geom[0] in self.robot.static_geoms_id and
                    not cnt.geom[1] in self.robot.mj_geom_eeff_id)
                    or
                    (cnt.geom[1] in self.robot.static_geoms_id and
                    not cnt.geom[0] in self.robot.mj_geom_eeff_id)
                    ):
                    cnt_pos_W = cnt.pos
                    q, _ = self.robot.get_pin_state()
                    T_b_W = pin.XYZQUATToSE3(q[:7]).inverse()
                    cnt_pos_b = T_b_W * cnt_pos_W
                    
                    id = self._get_collision_id(cnt_pos_b)
                    collision_contacts[id] = 1
                    break # Only one contact
                
        if np.sum(collision_contacts) == 0.:
            collision_contacts[-1] = 1
                
        return collision_contacts
    
    def record_failure(self) -> None:
        # Last jump is failure
        if len(self.record_state) > 0:
            # State
            current_state = np.zeros_like(self.record_state[-1])
            self.record_state.append(current_state)
            # Contact
            contact_locations_w = np.zeros_like(self.record_feet_contact[-1])
            self.record_feet_contact.append(contact_locations_w)
            # Target
            if self.i_jump + 1 > len(self.contact_plan) - 1:
                target_location_w = np.array(self.contact_plan[-1])
            else:
                target_location_w = self.contact_plan[self.i_jump + 1]
            self.record_target_contact.append(target_location_w)
            # Contacts           
            collision_contacts = self._check_collision()
            self.record_collision.append(collision_contacts)
            # self.record_collision[-1] = collision_contacts

    def record(self, q: np.array, v: np.array, robot_data: MjData) -> None:
        self._update_consecutive_landing()

        if (self.consecutive_landing > RecordJumpData.MIN_STEP_IN_CONTACT and
            self.waiting_for_next_jump):
            
            # State
            current_state = np.concatenate((q, v), axis=0)
            self.record_state.append(current_state)
                        
            # Contact
            contact_locations_w = self.robot.get_pin_feet_position_world()
            self.record_feet_contact.append(contact_locations_w)
            
            # Target
            self.i_jump = int((robot_data.time + jump.gait_horizon / 2) // (jump.gait_horizon / 2))
            if self.i_jump > len(self.contact_plan) - 1:
                target_location_w = np.array(self.contact_plan[-1])
            else:
                target_location_w = np.array(self.contact_plan[self.i_jump])
            self.record_target_contact.append(target_location_w)

            # Contacts
            self.record_collision.append(np.zeros(5, dtype=np.int8)) # 0 When no collision
            
            self.waiting_for_next_jump = False
    
    def _append_and_save(self, skip_first, skip_last):
        # if len(self.record_state) - skip_first - skip_last > 0:
        if len(self.record_state) > 0:

            # Skip first and last
            N = len(self.record_state)
            self.record_state = self.record_state[skip_first:N-skip_last]
            self.record_feet_contact = self.record_feet_contact[skip_first:N-skip_last]
            self.record_target_contact = self.record_target_contact[skip_first:N-skip_last]
            self.record_collision = self.record_collision[skip_first:N-skip_last]
            
            # Load data if exists
            if os.path.exists(self.saving_file_path):
                data = np.load(self.saving_file_path)
                record_state = data[RecordJumpData.STATE_NAME]
                record_feet_contact = data[RecordJumpData.CONTACT_NAME]
                record_target_contact = data[RecordJumpData.TARGET_NAME]
                record_collision = data[RecordJumpData.COLLISION_NAME]
                
                self.record_state = np.concatenate((record_state, self.record_state), axis=0)
                self.record_feet_contact = np.concatenate((record_feet_contact, self.record_feet_contact), axis=0)
                self.record_target_contact = np.concatenate((record_target_contact, self.record_target_contact), axis=0)
                self.record_collision = np.concatenate((record_collision, self.record_collision), axis=0)
            
            # Save with new data
            d = {
                RecordJumpData.STATE_NAME : self.record_state,
                RecordJumpData.CONTACT_NAME : self.record_feet_contact,
                RecordJumpData.TARGET_NAME : self.record_target_contact,
                RecordJumpData.COLLISION_NAME : self.record_collision,
            }
            
            np.savez(self.saving_file_path, **d)
    
    def save(self, skip_first : int = 0, skip_last : int = 0, lock = None) -> None:
        if lock:
            with lock:
                self._append_and_save(skip_first, skip_last)
        else:
            self._append_and_save(skip_first, skip_last)

        
def create_random_walk_contact_plan(current_feet_pos : np.ndarray,
                                    v : float = V,
                                    length : int = LENGTH,
                                    change_dir_step : int = CHANGE_DIR_STEP,
                                    scale_noise : float = SCALE_NOISE,
                                    seed : int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    contact_plan = [current_feet_pos]
    moving_offset = np.zeros_like(current_feet_pos)
    N_eeff = current_feet_pos.shape[0]
    for i in range(length):
        
        # Randomized next position
        # if i % change_dir_step == 0:
        #     # dir_x, dir_y = np.tile(np.random.randint(-1, 2), 4), np.tile(np.random.randint(-1, 2), 4)
        #     dir_x, dir_y = np.tile(1.0, 4), np.tile(0.0, 4)
        #     #dir_x, dir_y = np.random.randint(-1, 2, size=(N_eeff)), np.random.randint(-1, 2, size=(N_eeff))


        dir_x = np.random.uniform(-0.01, 0.4)
        dir_y = np.random.uniform(-0.3, 0.3)
        dir_x, dir_y = np.tile(dir_x, 4), np.tile(dir_y, 4)
        offset = np.array([dir_x, dir_y, np.zeros(N_eeff)]).T
        # offset_x = np.random.choice([-0.18, 0, 0.18], 4, p=[0.1, 0.4, 0.5])
        # offset_y = np.random.choice([-0.14, 0, 0.14], 4, p=[0.2, 0.6, 0.2])

        # offset = np.stack([offset_x, offset_y, np.zeros(N_eeff)], axis=-1)

        moving_offset += offset
        # moving_offset = np.mean(contact_plan[-1], axis=0)
        # offset[:, :2] += np.random.randn(4, 2) * scale_noise
        
        # Append to contact plan
        # next_contact = contact_plan[-1] + offset
        next_contact = current_feet_pos + moving_offset
        next_contact[:, :2] += np.random.randn(4, 2) * scale_noise
        contact_plan.append(next_contact.tolist())
        
    return contact_plan

def record_data(robot_source: QuadrupedWrapperAbstract,
                saving_path: str,
                seed : int = 0,
                queue : Queue = None,
                lock = None):

    robot = copy.copy(robot_source)
    robot.reset()
    feet_pos_w = robot.get_pin_feet_position_world()
    contact_plan = create_random_walk_contact_plan(feet_pos_w, seed=seed)

    ### Record data
    jump_recorder = RecordJumpData(robot, contact_plan, saving_path)

    ### Controller
    controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False)
    controller.set_gait_params(jump)  # Choose between trot, jump and bound
    controller.set_contact_plan(copy.deepcopy(contact_plan))

    ### Simulator
    simulator = Simulator(
        robot,
        controller,
        data_recorder=jump_recorder)
    
    if saving_path == "":
        def visual_callback(viewer, step, q, v, data) : 
            i = int(data.time // (jump.gait_horizon / 2))
            i_next = int((data.time + jump.gait_horizon / 2) // (jump.gait_horizon / 2))
            current_target = contact_plan[i]
            next_target = contact_plan[i_next]
            
            feet_pos_w = robot.get_pin_feet_position_world()
            positions = np.concatenate((current_target, next_target, feet_pos_w), axis=0)
            position_3d_callback(viewer, positions)
        
        simulator.run(use_viewer=True, verbose=True, stop_on_collision=True, visual_callback_fn=visual_callback)
        # print(jump_recorder._check_collision())
    else:
        simulator.run(use_viewer=False, verbose=False, stop_on_collision=True)
        jump_recorder.record_failure()
        jump_recorder.save(skip_first=0, lock=lock)
    
    if queue != None:
        queue.put(1)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Argument parser for simulation parameters.")
    parser.add_argument('--saving_path', type=str, default=DEFAULT_PATH, help='Path to save data')
    parser.add_argument('--N', type=int, default=10, help='Number of elements')
    parser.add_argument('--cores', type=int, default=10, help='Number of cores')
    parser.add_argument('--scale', type=float, default=SCALE_NOISE, help='Scaling factor for randomized position')
    parser.add_argument('--dir_step', type=int, default=CHANGE_DIR_STEP, help='Epsilon value')
    parser.add_argument('--v', type=float, default=V, help='Velocity scale')
    parser.add_argument('--test', action="store_true", help='Simulation with visualization.')

    args = parser.parse_args()
    
    # Configuration and paths
    cfg = Go2Config  # Assuming Go2Config is defined elsewhere

    URDF_path, xml_string, package_dir = RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir, models_path="./robots")

    # Load robot
    robot = QuadrupedWrapperAbstract(
        URDF_path,
        xml_string,
        package_dir,
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        foot_size=cfg.foot_size,
    )
    
    if args.test:
        record_data(robot, "")
        
    else:
    
        # def multiprocess_record_data(saving_path, seed, queue, lock) -> None:
        #     r = copy.copy(robot)
        #     record_data(r, saving_path, seed, queue, lock)
        
        # manager = Manager()
        # lock = manager.Lock()
        # queue = manager.Queue()
        # seeds = np.random.randint(0, 2**32 - 1, size=args.N)

        # tasks = [(args.saving_path, seed, queue, lock) for seed in seeds]
        
        # def worker(args):
        #     saving_path, seed, queue, lock = args
        #     multiprocess_record_data(saving_path, seed, queue, lock)

        # # Use multiprocessing Pool to run record_data in parallel
        # with Pool(processes=args.cores) as pool:
        #     # Wrap the pool.map with tqdm to display the progress bar
        #     for _ in tqdm(pool.imap_unordered(worker, tasks), total=args.N):
        #         queue.get()

        for _ in tqdm(range(args.N)):
            record_data(robot, args.saving_path, seed=np.random.randint(0, 2**32 - 1))