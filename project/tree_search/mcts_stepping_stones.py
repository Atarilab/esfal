import copy
import numpy as np
from numpy import sqrt
import tqdm
import time
from typing import List
from itertools import product, chain

from environment.sim import SteppingStonesSimulator
from tree_search.kinematics import QuadrupedKinematicFeasibility
from tree_search.abstract import MCTS, timing
from learning_jump_feasibility.test.test_utils import load_model, is_feasible, predict_next_state

# State is the current 4 contact locations, referenced by their indices
State = List[int]

class MCTSSteppingStonesKin(MCTS):
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
            avg_d_goal = MCTSSteppingStonesKin.avg_dist_to_goal(
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

class MCTSSteppingStonesDyn(MCTSSteppingStonesKin):
    def __init__(self,
                 stepping_stones_sim: SteppingStonesSimulator,
                 simulation_steps: int = 1,
                 C: float = np.sqrt(2),
                 W: float = 10,
                 alpha_exploration: float = 0,
                 state_estimator_state_path: str = "",
                 classifier_state_path: str = "",
                 classifier_threshold: float = 0.9,
                 safety: float = 1.,
                 **kwargs
                 ) -> None:
        
        super().__init__(stepping_stones_sim,
                         simulation_steps,
                         C,
                         W,
                         alpha_exploration,
                         **kwargs)
        
        # Estimate the state at next jump + input of MPC to reach that position
        self.state_estimator = load_model(state_estimator_state_path)
        # Classify if a jump dynamically feasible
        self.classifier = load_model(classifier_state_path)
        # Threshold on the probability of success to consider a jump dyn feasible
        self.classifier_threshold = classifier_threshold
        self.safety = safety
        # Store the predicted states for each state path, full state {hash : [q, v]}
        self.node_state_dict = {}
        # Store the score associated to each state path, {hash : score}
        self.node_score_dict = {}
    
    @timing("selection")
    def selection(self, state: List[int], state_goal: List[int]) -> List[int]:
        selected_state = super().selection(state, state_goal)
        self.predict_robot_state()
        return selected_state
    
    @staticmethod
    def remove_last_repeated_elements(lst):
        if len(lst) < 3:
            return lst  # Return the list as is if it has less than 3 elements
        
        last_elem = lst[-1]
        count = 0
        
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] == last_elem:
                count += 1
                if count > 2:
                    lst.pop(i)
            else:
                break
                
        return lst

    @staticmethod
    def remove_repeated_elements(lst):
        # Dictionary to store the counts of each sublist
        counts = {}
        
        # Create a new list to store the result
        result = []
        
        for sublist in lst:
            # Convert the sublist to a tuple (which is hashable) for counting purposes
            sublist_tuple = tuple(sublist)
            
            # Update the count in the dictionary
            if sublist_tuple not in counts:
                counts[sublist_tuple] = 0
            
            if counts[sublist_tuple] < 2:
                result.append(sublist)
                counts[sublist_tuple] += 1

        return result

    def get_full_current_path(self, next_state : State = []) -> List[int]:
        """
        Get full current contact path a single non nested list.
        Don't allow repetition of more than two elements/
        Args:
            next_state (State, optional): state to add to the path.
                Defaults to [].
        Returns:
            List[int]: List of int.
        """
        full_current_path = list(chain(*self.remove_repeated_elements(self.tree.current_search_path)))
        full_current_path += list(next_state)
        return full_current_path
    
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
        h_distance = self.avg_dist_to_goal(
            self.sim.stepping_stones.positions,
            states,
            goal_state)

        # TODO Batched version
        h_feasible = np.empty_like(h_distance)
        for i, state in enumerate(states):
            full_path = self.get_full_current_path(state)
            h_state = self.tree.hash_state(full_path)
            h_feasible[i] = self.node_score_dict.get(h_state, 0.)
        
        heuristic_values = h_distance + self.safety * h_feasible
        
        # Exploration
        if np.random.rand() < self.alpha_exploration:
            probs = heuristic_values / sum(heuristic_values)
            id = np.random.choice(np.arange(len(states)), p=probs)

        # Exploitation
        else:
            id = np.argmin(heuristic_values)
        
        state = states[id]

        return state
        
    def predict_robot_state(self) -> np.ndarray:
        """
        Return the robot state at the end of the current contact plan.
        Compute it with the regressor if not stored.

        Returns:
            np.ndarray: state predicted by the regressor
        """
        full_path = self.get_full_current_path()
        h_full_path = self.tree.hash_state(full_path)
        if h_full_path in self.node_state_dict:
            robot_state = self.node_state_dict[h_full_path]
            q, v = np.split(robot_state, [7 + 12])
            return q, v
    
        else:
            target_contact_id = full_path[-4:]
            # First jump
            if len(full_path) < 8:
                q, v = self.sim.robot.get_pin_state()
                current_contact_id = full_path[-4:]
            # At least two jumps
            else:
                # State before last jump
                last_path = full_path[:-4]
                h_last_path = self.tree.hash_state(last_path)
                last_robot_state = self.node_state_dict[h_last_path]
                q, v = np.split(last_robot_state, [7 + 12])
                current_contact_id = full_path[-8:-4]
            
            target_contact_w = self.sim.stepping_stones.positions[target_contact_id]
            current_contact_w = self.sim.stepping_stones.positions[current_contact_id]

            q, v, _ = predict_next_state(self.state_estimator, q, v, self.sim.robot, current_contact_w, target_contact_w)
            robot_state = np.concatenate((q, v))
            self.node_state_dict[h_full_path] = robot_state
            
        return q, v
            
    def is_dynamically_feasible(self, state: State, next_states: List[State]) -> np.ndarray:
        """
        Predict which states are dynamically feasible.

        Args:
            state (List[State]): Current state.
            next_states (List[State]): Potential next states.

        Returns:
            np.ndarray: bool array. True if feasible.
        """
        B = len(next_states)

        ### Expressing the network inputs
        # State (should be already estimated in the last expansion phase)   
        q, v = self.predict_robot_state()
        
        # Current feet locations
        current_pos_w = self.sim.stepping_stones.positions[state].reshape(1, -1)
        current_pos_batched_w = np.repeat(current_pos_w, B, axis=0)
        
        # Target contact locations
        target_pos_w = self.sim.stepping_stones.positions[next_states].reshape(B, -1)

        ### Predict feasibility with classifier in a batch way
        dyn_feasible, proba_success = is_feasible(self.classifier,
                                                  q,
                                                  v,
                                                  current_pos_batched_w,
                                                  target_pos_w,
                                                  self.classifier_threshold)

        # TODO batch version
        # Save the probability of success, used in the heuristic
        for next_state, feasible, prob in zip(next_states, dyn_feasible, proba_success):
            if feasible:
                full_next_path = self.get_full_current_path(next_state)
                h_full_next_path = self.tree.hash_state(full_next_path)
                self.node_score_dict[h_full_next_path] = prob.item()
        
        return dyn_feasible
        
    def get_children(self, state: List[int]) -> List[List[int]]:
        """
        Get kinematically and dynamically reachable states
        from the current state.

        Args:
            state (State): current state.

        Returns:
            List[State]: Reachable states as a list.
        """
        feet_pos_w = self.sim.stepping_stones.positions[state[-4:]]

        # Shape [N_reachable, 4]
        possible_contact_id = [
            self.kinematics.reachable_locations(
            foot_pos,
            self.sim.stepping_stones.positions,
            scale_reach=0.55 # contact above this distance will not be considered
            ) for foot_pos in feet_pos_w]

        # Combinaison of feet location [N_comb, 4]
        possible_next_contact = np.array(list(product(*possible_contact_id)))

        # Legs not crossed, easy pruning
        feasible = self.kinematics._check_cross_legs(self.sim.stepping_stones.positions[possible_next_contact])
        # Dynamic feasibility
        dyn_feasible = self.is_dynamically_feasible(state, possible_next_contact[feasible])

        feasible[feasible] = dyn_feasible
        legal_next_states = possible_next_contact[feasible]
            
        return legal_next_states