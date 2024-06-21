
from typing import Any, Callable, Tuple
import numpy as np
from environment.stepping_stones import SteppingStonesEnv
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.bicon_mpc import BiConMPC
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract

class SteppingStonesSimulator(Simulator):
    """
    Simulator with stepping stones environment.
    """
    
    # Maximum average distance between feet and stones to accept a randomly
    # selected set of 4 contact locataions 
    MAX_DIST_RANDOM_LOCATION = 0.07 # m
    # Height offset when initialising start position
    HEIGHT_OFFSET_START = 0.03 # m
    # Minimun number of steps the goal is reached consecutively
    MIN_GOAL_CONSECUTIVE = 6
    # Check if robot reached goal every <CHECK_GOAL_PERIOD> steps
    CHECK_GOAL_PERIOD = 150
    
    def __init__(self,
                 stepping_stones_env: SteppingStonesEnv,
                 robot: QuadrupedWrapperAbstract,
                 controller: BiConMPC,
                 data_recorder: DataRecorderAbstract = None,
                 **kwargs) -> None:
        super().__init__(robot, controller, data_recorder)
        
        self.stepping_stones = stepping_stones_env
        self.feet_pos_0 = self.robot.get_pin_feet_position_world()
        self.q0, _ = self.robot.get_mj_state()
        self.start_pos = []
        self.start_indices = []
        self.goal_pos = []
        self.goal_indices = []
        self.consec_on_goal = 0
        
        optionals = {
            "min_goal_consecutive" : SteppingStonesSimulator.MIN_GOAL_CONSECUTIVE,            
            "height_offset" : SteppingStonesSimulator.HEIGHT_OFFSET_START,            
        }
        optionals.update(kwargs)
        for k, v in optionals.items(): setattr(self, k, v)
        
    def _sample_random_feet_stones_locations(self,
                                             max_dist_to_center: float = -1) -> list[int]:
        """
        Returns possible stones positions for the feet
        based on the nominal configuration.
        
        Args:
            max_dist_to_center (float): Maximum distance to center of the env.

        Returns:
            list[int]: indices of the 4 selected stepping stones (list)
        """
        i = 0
        position_found = False
        
        while not position_found or i > self.stepping_stones.N:
            
            # Get center stone id
            
            # Sample a random stone id for the center
            if max_dist_to_center < 0:
                id_stone_center = np.randint(self.stepping_stones.N)
            # Take the stepping stone in the center
            elif max_dist_to_center == 0:
                id_stone_center, _ = self.stepping_stones.get_closest(np.zeros((1, 3)))
            # Draw at random among the possible ones
            else:
                random_pos = np.random.rand(1, 3) * max_dist_to_center
                random_pos[:, 2] = self.stepping_stones.height
                id_stone_center, _ = self.stepping_stones.get_closest(random_pos)
                            
            # Check if feet in nominal position have stones closeby
            
            feet_pos_0_centered = self.feet_pos_0 - np.mean(self.feet_pos_0, axis=0, keepdims=True)
            stone_center_pos = self.stepping_stones.positions[id_stone_center]
            feet_pos = stone_center_pos + feet_pos_0_centered
            closest_stones_id, closest_stones_pos = self.stepping_stones.get_closest(feet_pos)
            avg_distance = np.mean(np.linalg.norm(closest_stones_pos - feet_pos))
            
            if avg_distance < SteppingStonesSimulator.MAX_DIST_RANDOM_LOCATION:
                position_found = True
                
            i += 1
                
        assert position_found, "Random feet position not found."
                
        return closest_stones_id
            
    def set_start_and_goal(self,
                           start_indices: np.ndarray | list[int] | Any = [],
                           goal_indices: np.ndarray | list[int] | Any = [],
                           max_dist: float = 1.,
                           init_robot_pos: bool = True) -> None:
        """
        Set the start and goal locations.
        If not provided, set them at random so that the start is at the center
        and the goal at a maximum distance from the center.
        """
        # Start stones
        if ((isinstance(start_indices, list) or
             isinstance(start_indices, np.ndarray)) and
            len(goal_indices) == 4):
            self.start_indices = start_indices
        else:
            self.start_indices = self._sample_random_feet_stones_locations(0.)
        
        # Goal stones
        if ((isinstance(goal_indices, list) or
             isinstance(goal_indices, np.ndarray)) and
            len(goal_indices) == 4):
            self.goal_indices = goal_indices
        else:
            self.goal_indices = self._sample_random_feet_stones_locations(max_dist)
        
        # Init positions
        self.start_pos = self.stepping_stones.positions[self.start_indices]
        self.goal_pos = self.stepping_stones.positions[self.goal_indices]
        
        # Set robot start state
        if init_robot_pos:
            q_start = self.q0.copy()
            q_start[:2] = np.mean(self.start_pos, axis=0)[:2]
            q_start[2] += self.stepping_stones.height + SteppingStonesSimulator.HEIGHT_OFFSET_START
            self.robot.reset(q_start)

            # Move start positions under the feet
            self.stepping_stones.set_start_position(self.start_pos)
        
    def _on_goal(self) -> bool:
        """
        Checks if the robot is on the goal locations.

        Returns:
            bool: True if on goal.
        """
        # {eeff_name : id(static_geom_name)}
        eeff_in_contact_floor = {}
        on_goal = False
        
        # Filter contacts
        for cnt in self.robot.mj_data.contact:
            if (cnt.geom[0] in self.robot.static_geoms_id and
                cnt.geom[1] in self.robot.mj_geom_eeff_id):
                eeff_name = self.robot.mj_model.geom(cnt.geom[1]).name
                static_geom_name = self.robot.mj_model.geom(cnt.geom[0]).name

                
            elif (cnt.geom[1] in self.robot.static_geoms_id and
                cnt.geom[0] in self.robot.mj_geom_eeff_id):
                eeff_name = self.robot.mj_model.geom(cnt.geom[0]).name
                static_geom_name = self.robot.mj_model.geom(cnt.geom[1]).name
        
            # Get the id of the stone from the geometry name
            try:
                static_geom_id = int(static_geom_name.replace(f"{self.stepping_stones.DEFAULT_GEOM_NAME}_", ""))
            except:
                continue
            
            eeff_in_contact_floor[eeff_name] = static_geom_id
        
        # Check all feet in contact are on goal location
        if set(self.goal_indices) == set(eeff_in_contact_floor.values()):
            on_goal = True
            self.consec_on_goal += 1
            
        elif len(eeff_in_contact_floor.values()) > 0:
            self.consec_on_goal = 0

        return on_goal
    
    def _simulation_step(self) -> None:
        super()._simulation_step()
        
        if self.sim_step % SteppingStonesSimulator.CHECK_GOAL_PERIOD == 0:
            self._on_goal()
            if self.consec_on_goal >= self.min_goal_consecutive:
                self.stop_sim = True
        
    def run_contact_plan(self,
                         contact_plan_id: np.ndarray,
                         use_viewer: bool = False,
                         visual_callback_fn: Callable = None,
                         **kwargs,
                         ) -> int:
        """
        Run simulation and controller with a given contact plan.

        Args:
            - contact_plan_id (np.ndarray): Indices of the contact locations. Shape [L, Neeff, 1].
            - use_viewer (bool, optional): Use viewer. Defaults to False.
            - visual_callback_fn (fn): function that takes as input:
                - the viewer
                - the simulation step
                - the state
                - the simulation data
            that create visual geometries using the mjv_initGeom function.
            See https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
            for an example.
        Returns:
            int: 1 if goal reached else 0.
        """
        verbose = kwargs.get("verbose", False)
        init_robot_pos = kwargs.get("init_robot_pos", True)
        
        self.set_start_and_goal(contact_plan_id[0], contact_plan_id[-1], init_robot_pos=init_robot_pos)
        self.consec_on_goal = 0
        
        contact_plan_pos = self.stepping_stones.positions[contact_plan_id]
        self.controller.set_contact_plan(contact_plan_pos)

        super().run(use_viewer=use_viewer,
                    visual_callback_fn=visual_callback_fn,
                    verbose=verbose,
                    stop_on_collision=True,
                    **kwargs
                    )
        
        goal_reached = (
            not(self.robot.collided) and
            self.consec_on_goal >= SteppingStonesSimulator.MIN_GOAL_CONSECUTIVE
            )

        return int(goal_reached)
            