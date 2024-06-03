import copy
import numpy as np
import tqdm
import time
from typing import List
from itertools import product

from environment.sim import SteppingStonesSimulator
from tree_search.kinematics import QuadrupedKinematicFeasibility
from tree_search.abstract import MCTS

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

        return reward, solution_found
               

# class MCTSSteppingStones(MCTS):
    
#     def __init__(self,
#                  stepping_stones_locations,
#                  record_data_dir:str,
#                  simulation_steps:int=1,
#                  alpha_exploration:float=0.1,
#                  C:float=0.03,
#                  sigma=None,
#                  W:float=1.,
#                  **kwargs,
#                  ) -> None:
        
#         self.stepping_stones_env = stepping_stones_env
#         self.record_data_dir = record_data_dir
#         self.alpha_exploration = alpha_exploration
#         self.W = W

#         self.contact_loc_w = stepping_stones_env.box_location
#         self.d_max = np.linalg.norm(self.contact_loc_w[0, :2] - self.contact_loc_w[-1, :2])

#         optional_args = {
#             "max_depth_selection" : 12,
#             "max_solution_search" : 1,
#             "replay_simulation" : 1,
#         }

#         self.performance = {
#             "time_first" : 0.,
#             "n_nmpc_first" : 0,
#             "iteration_first" : 0,
#         }

#         optional_args.update(kwargs)

#         super().__init__(reward_fn=self.opposit_avg_distance_to_goal,
#                          get_children_fn=self.get_kinematically_reachable_states,
#                          simulation_steps=simulation_steps,
#                          C=C,
#                          **optional_args)
        
#         # TODO Try to fix when sigma is passed in arg
#         T = 8
#         self.sigma = lambda reward :  1. / (np.exp(-T * (reward - 1.)) + 1.)

#     def opposit_avg_distance_to_goal(self, state, goal_state):
#         legs_w = self.contact_loc_w[state]
#         goals_w = self.contact_loc_w[goal_state]
#         avg_dist_to_goal = np.mean(np.linalg.norm(legs_w - goals_w, 2, axis=-1))
#         reward = 1. - avg_dist_to_goal / self.d_max
#         return reward
    
#     def get_all_close_box_id(self, current_box_location_w, max_dist = .2):
#         dist = np.linalg.norm((np.expand_dims(current_box_location_w[:2], 0) - self.contact_loc_w[:, :2]), axis=-1) # [N, 1]
#         dist[(self.contact_loc_w[:, 2] == 0.)] += 2 * max_dist
#         id = np.where(dist < max_dist)[0].tolist()
#         return id

#     def get_kinematically_reachable_states(self, state):
#         # Reject contact locations too far away
#         leg_pos_w =  [self.contact_loc_w[i, :] for i in state]
#         possible_next_contact = [self.get_all_close_box_id(pos) for pos in leg_pos_w]

#         #TODO : Try with actual kinematic check 
#         def legs_not_crossing(leg_pos_id):
#             FL  = leg_pos_id[0]
#             HL  = leg_pos_id[1]
#             FR  = leg_pos_id[2]
#             HR  = leg_pos_id[3]

#             # Legs can't cross
#             # Assuming robot orientation is fixed with
#             # 
#             #          HL ---- FL
#             #  ^ y       ##### ->
#             #  !       HR ---- FR
#             #  !--> x 
#             # 
#             # And id of boxes are
#             # 2L ....
#             # L   L+1  L+2  ...
#             # 0    1    2    3  ....   L
#             #

#             if FL < HL or \
#                FR < HR or \
#                FL < FR or \
#                HL < HR :

#                 return True
#             return False
        
#         legal_next_states = list(filterfalse(legs_not_crossing, product(*possible_next_contact)))
#         return legal_next_states
    
#     def nmpc_simulation(self, contact_plan):
        
#         contact_plan += [contact_plan[-1]] * 5 # Jump several on the goal location
#         position_ini = self.stepping_stones_env.get_start_position()

#         if self.performance["iteration_first"] == 0:
#             self.performance["n_nmpc_first"] += 1

#         for step in range(self.replay_simulation + 1):

#             self.simulator = Solo12Simulator(self.record_data_dir, False, False, randomized=step > 0) # Randomized for replay

#             success = self.simulator.run(
#                 model=None,
#                 contact_plan=contact_plan,
#                 box_location=self.contact_loc_w,
#                 position_ini=position_ini,
#                 server=DIRECT,
#             )

#             if step == 0:
#                 print("Simulation NMPC succeed:", success)
#                 if not success:
#                     return -self.W

#         return self.W
    
#     def simulation(self, state, goal_state) -> float:
#         simulation_path = []

#         feasible_solution_found = False
#         reward = 0.

#         for _ in range(self.simulation_steps):
#             if self.tree.has_children(state) and not self.is_terminal(state, goal_state):
                
#                 children = self.tree.get_children(state)
#                 children_reward = list(map(lambda state : self.reward(state, goal_state), children))
#                 # Exploration
#                 if np.random.rand() < self.alpha_exploration:
#                     probs = children_reward / sum(children_reward)
#                     id_next_state = np.random.choice(np.arange(len(children)), p=probs)

#                 # Exploitation
#                 else:
#                     id_next_state = np.argmax(children_reward)

#                 state = children[id_next_state]
#                 reward = children_reward[id_next_state]
            
#                 simulation_path.append(state)

#         W = self.W
#         if self.is_terminal(state, goal_state):
#             contact_plan = self.tree.current_search_path + simulation_path
#             W = self.nmpc_simulation(contact_plan)
#             feasible_solution_found = W > 0

#         reward *= W
        
#         return reward, feasible_solution_found
    
#     def search(self, num_iterations:int=1000):
#         state_start = list(self.stepping_stones_env.get_id_feet_start())
#         state_goal = list(self.stepping_stones_env.get_id_feet_goal())
#         progress_bar = tqdm.trange(0, num_iterations)
#         self.solution_found = 0

#         self.tree.reset_search_path()

#         search_time = time.time()
#         for it in progress_bar:
#             # Selection
#             leaf = self.selection(state_start)
#             # Expansion
#             self.expansion(leaf)
#             # Simulation
#             reward, terminal_state = self.simulation(leaf, state_goal)
#             # Backpropagation
#             self.back_propagation(reward)

#             if self.print_info and it % 10 == 0:
#                 progress_bar.set_postfix({
#                         "nodes": len(self.tree.nodes),
#                         "reward": reward})

#             if terminal_state:
#                 self.solution_found += 1

#                 if self.solution_found == 1:
#                     self.performance["time_first"] = time.time() - search_time
#                     self.performance["it_first"] = it
                
#                 if self.solution_found >= self.max_solution_search:
#                     break

#     def get_best_children(self, n_children: int = 10, mode: str = "value") -> List:
#         state_start = self.stepping_stones_env.start_feet
#         state_goal = self.stepping_stones_env.goal_feet
#         return super().get_best_children(state_start, state_goal, n_children, mode)
    
#     def get_perfs(self):
#         return self.performance