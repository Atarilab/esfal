import copy
import numpy as np
from numpy import sqrt
import tqdm
import time
from typing import List
from itertools import product

from environment.sim import SteppingStonesSimulator
from tree_search.kinematics import QuadrupedKinematicFeasibility
from tree_search.abstract import MCTS, timing

State = List[int]

class MCTSSteppingStones(MCTS):
    def __init__(self,
                 stepping_stones_sim: SteppingStonesSimulator,
                 simulation_steps: int = 1,
                 C: float = np.sqrt(2),
                 W: float = 10.,
                 alpha_exploration: float = 0.0,
                 **kwargs,
                 ) -> None:
        
        self.sim = stepping_stones_sim
        self.alpha_exploration = alpha_exploration
        self.C = C
        self.W = W

        optional_args = {
            "max_depth_selection" : 12,
            "max_solution_search" : 3,
            "n_threads_kin" : 10,
            "n_threads_sim" : 10,
        }
        optional_args.update(kwargs)

        self.performance = {
            "time_first" : 0.,
            "n_nmpc_first" : 0,
            "iteration_first" : 0,
        }
        
        super().__init__(simulation_steps, C, **optional_args)
        
        # Maximum distance between contact locations
        self.d_max = self._compute_max_dist(self.sim.stepping_stones.positions)
        
        # Kinematics feasibility
        self.kinematics = QuadrupedKinematicFeasibility(self.sim.robot, num_threads=self.n_threads_kin)
            
    def _compute_max_dist(self, contact_pos_w) -> float:
        diffs = contact_pos_w[:, np.newaxis, :] - contact_pos_w[np.newaxis, :, :]
        d_squared = np.sum(diffs**2, axis=-1)
        d_max = np.sqrt(np.max(d_squared))
        return d_max

    @staticmethod
    def avg_dist_to_goal(contact_pos_w: np.ndarray,
                        current_states: list[State],
                        goal_state: State) -> float:
        """
        Computes average distance to goal.
        """
        d_to_goal = contact_pos_w[current_states] - contact_pos_w[np.newaxis, goal_state]
        avg_dist_to_goal = np.mean(np.linalg.norm(d_to_goal, axis=-1), axis=-1)
        return avg_dist_to_goal
    
    @timing("heuristic")
    def heuristic(self,
                  states: list[State],
                  goal_state: State) -> State:
        """
        Heuristic function to guide the search computed in a batched way.
        
        Args:
            states (List[State]): Compute the value of the heuristic on those states.
            goal_state (State): Goal state.

        Returns:
            State: State chosen by the heuristic.

        """
        heuristic_values = self.avg_dist_to_goal(
            self.sim.stepping_stones.positions,
            states,
            goal_state)

        # Exploration
        if np.random.rand() < self.alpha_exploration:
            probs = heuristic_values / sum(heuristic_values)
            id = np.random.choice(np.arange(len(states)), p=probs)

        # Exploitation
        else:
            id = np.argmin(heuristic_values)
        
        state = states[id]
        return state
    
    def get_children(self, state: State) -> List[State]:
        """
        Get kinematically reachable states from the current state.

        Args:
            state (State): current state.

        Returns:
            List[State]: Reachable states as a list.
        """
        feet_pos_w = self.sim.stepping_stones.positions[state]

        # Shape [Nr, 4]
        possible_contact_id = [
            self.kinematics.reachable_locations(
            foot_pos,
            self.sim.stepping_stones.positions,
            scale_reach=0.55
            ) for foot_pos in feet_pos_w]

        # Combinaison of feet location [NComb, 4]
        possible_states = np.array(list(product(*possible_contact_id)))

        # Bool array [NComb]
        reachable = self.kinematics.is_feasible(
            self.sim.stepping_stones.positions[possible_states],
            allow_crossed_legs=False,
            check_collision=True
            )
        
        legal_next_states = possible_states[reachable]
        return legal_next_states

    @staticmethod
    def reward(contact_plan: list[list[State]],
               goal_state: State,
               d_max: float,
               W: float,
               sim: SteppingStonesSimulator,
               ) -> float:
        
        if contact_plan[-1] != goal_state:
            avg_d_goal = MCTSSteppingStones.avg_dist_to_goal(
                sim.stepping_stones.positions,
                contact_plan[-1],
                goal_state,
            )[0]
            return 1 - avg_d_goal / d_max
        
        goal_reached = sim.run_contact_plan(contact_plan) * W
        return goal_reached
    
    @timing("simulation")
    def simulation(self, state, goal_state) -> float:
        
        simulation_path = []
        for _ in range(self.simulation_steps):
            
            # Choose successively one child until goal is reached
            if self.tree.has_children(state) and not self.is_terminal(state, goal_state):
                children = self.tree.get_children(state)
                state = self.heuristic(children, goal_state)

                simulation_path.append(state)
            else:
                break
        
        contact_plan = self.tree.current_search_path + simulation_path
        reward = self.reward(contact_plan, goal_state, self.d_max, self.W, self.sim)
        solution_found = reward >= 1
        
        if solution_found:
            self.solutions.append(contact_plan)

        return reward, solution_found