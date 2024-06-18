import copy
import numpy as np
from numpy import sqrt
import torch
import tqdm
import time
from typing import List
from itertools import product

from environment.sim import SteppingStonesSimulator
from tree_search.kinematics import QuadrupedKinematicFeasibility
from tree_search.mcts import MCTSSteppingStones, timing

from learning_jump_feasibility.test.test_utils import load_model, is_feasible, predict_next_state

# State is the whole path from start to goal
State = List[int]
    
class MCTSSteppingStonesDyn(MCTSSteppingStones):
    def __init__(self,
                 stepping_stones_sim: SteppingStonesSimulator,
                 simulation_steps: int = 1,
                 C: float = np.sqrt(2),
                 W: float = 10,
                 alpha_exploration: float = 0,
                 state_estimator_state_path: str = "",
                 classifier_state_path: str = "",
                 classifier_threshold: float = 0.8,
                 prudence: float = 1.,
                 **kwargs
                 ) -> None:
        
        super().__init__(stepping_stones_sim,
                         simulation_steps,
                         C,
                         W,
                         alpha_exploration,
                         **kwargs)
        
        self.state_estimator = load_model(state_estimator_state_path)
        self.classifier = load_model(classifier_state_path)
        self.classifier_threshold = classifier_threshold
        self.prudence = prudence
        # Store the predicted states for each state path, full state {hash : [q, v]}
        self.node_state_dict = {}
        # Store the score associated to each state path, {hash : score}
        self.node_score_dict = {}
    
    @timing("selection")
    def selection(self, state: List[int], state_goal: List[int]) -> List[int]:
        selected_state = super().selection(state, state_goal)
        self.predict_robot_state(selected_state)
        return selected_state
    
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
        last_states = np.array(states)[:, -4:]

        h_distance = self.avg_dist_to_goal(
            self.sim.stepping_stones.positions,
            last_states,
            goal_state)

        # TODO Batched version
        h_feasible = np.empty_like(h_distance)
        for i, state in enumerate(states):
            h_state = self.tree.hash_state(state)
            h_feasible[i] = self.node_score_dict.get(h_state, 0.)
        
        heuristic_values = h_distance + self.prudence * h_feasible
        
        # Exploration
        if np.random.rand() < self.alpha_exploration:
            probs = heuristic_values / sum(heuristic_values)
            id = np.random.choice(np.arange(len(states)), p=probs)

        # Exploitation
        else:
            id = np.argmin(heuristic_values)
        
        state = states[id]

        return state
    
    @staticmethod
    def is_terminal(start_state : State, state_goal : State) -> bool:
        # Last contact on goal location
        return start_state[-4:] == state_goal
        
    def predict_robot_state(self, state: State) -> np.ndarray:
        """
        Return the robot state at the end of the current contact plan.
        Compute it with the regressor if not stored.

        Args:
            state (State): Current whole contact plan.
            next_states (State): Current whole contact plan.

        Returns:
            np.ndarray: state predicted by the regressor
        """
        h_state = self.tree.hash_state(state)
        if h_state in self.node_state_dict:
            return self.node_state_dict[h_state]
        
        else:
            target_contact_id = state[-4:]
            # First jump
            if len(state) < 8:
                q, v = self.sim.robot.get_pin_state()
                current_contact_id = state[-4:]
            # At least two jumps
            else:
                # State before last jump
                last_state = state[:-4]
                h_last_state = self.tree.hash_state(last_state)
                last_robot_state = self.node_state_dict[h_last_state]
                q, v = np.split(last_robot_state, [7 + 12])
                current_contact_id = state[-8:-4]
            
            target_contact_w = self.sim.stepping_stones.positions[target_contact_id]
            current_contact_w = self.sim.stepping_stones.positions[current_contact_id]

            q, v, _ = predict_next_state(self.state_estimator, q, v, self.sim.robot, current_contact_w, target_contact_w)
            robot_state = np.concatenate((q, v))
            self.node_state_dict[h_state] = robot_state
            
        return robot_state
            
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
        robot_state = self.predict_robot_state(state)
        q, v = np.split(robot_state, [7+12])
        
        # Current feet locations
        current_contact_id = state[-4:]
        current_pos_w = self.sim.stepping_stones.positions[current_contact_id].reshape(1, -1)
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
        # Save the proba success, eventually for the heuristic
        for next_state, feasible, prob in zip(next_states, dyn_feasible, proba_success):
            if feasible:
                full_state = state + [next_state]
                h_full_state = self.tree.hash_state(full_state)
                self.node_score_dict[h_full_state] = prob.item()
        
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

        # Shape [Nr, 4]
        possible_contact_id = [
            self.kinematics.reachable_locations(
            foot_pos,
            self.sim.stepping_stones.positions,
            scale_reach=0.55
            ) for foot_pos in feet_pos_w]

        # Combinaison of feet location [NComb, 4]
        possible_next_contact = np.array(list(product(*possible_contact_id)))

        # Legs not crossed, easy pruning
        feasible = self.kinematics._check_cross_legs(self.sim.stepping_stones.positions[possible_next_contact])

        # Dynamic feasibility
        dyn_feasible = self.is_dynamically_feasible(state, possible_next_contact[feasible])
        feasible[feasible] = dyn_feasible
        
        legal_next_contact = possible_next_contact[feasible]
        legal_next_states = [state + next_contact.tolist() for next_contact in legal_next_contact]

        return legal_next_states

    @staticmethod
    def reward(contact_plan: np.array,
               goal_state: State,
               d_max: float,
               W: float,
               sim: SteppingStonesSimulator,
               ) -> float:
        
        last_contact = contact_plan[-1].tolist()
        if last_contact != goal_state:
            avg_d_goal = MCTSSteppingStones.avg_dist_to_goal(
                sim.stepping_stones.positions,
                last_contact,
                goal_state,
            )[0]
            return 1 - avg_d_goal / d_max
        
        goal_reached = sim.run_contact_plan(contact_plan) * W

        return goal_reached
    
    @timing("simulation")
    def simulation(self, state, goal_state) -> float:
        
        for _ in range(self.simulation_steps):
            
            # Choose successively one child until goal is reached
            if self.tree.has_children(state) and not self.is_terminal(state, goal_state):
                children = self.tree.get_children(state)
                state = self.heuristic(children, goal_state)

            else:
                break
        
        contact_plan = np.array(state).reshape(-1, 4)

        reward = self.reward(contact_plan, goal_state, self.d_max, self.W, self.sim)
        solution_found = reward >= 1
        
        if solution_found >= 1:
            self.solutions.append(contact_plan)

        return reward, solution_found