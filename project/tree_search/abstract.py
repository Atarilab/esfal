import numpy as np
import tqdm
import time
from typing import List
from collections import defaultdict
from functools import wraps

class Action:
    def __init__(self) -> None:
        self.visit = 0
        self.value = 0.

    def increment_visit(self) -> None:
        self.visit += 1

    def update_value(self, reward) -> None:
        self.value += reward

class Node:
    def __init__(self) -> None:
        self.visit = 0
        self.actions = defaultdict(Action)             # { child state (hash): Action }
    
    def increment_visit(self) -> None:
        self.visit += 1

class Tree:
    def __init__(self) -> None:
        self.nodes = defaultdict(Node)               # { state (hash) : Node }

        self.current_search_path = []               # Path of the current search List[state]

    @staticmethod
    def hash_state(state:List[int]) -> str:
        hash_string = '-'.join(map(str, state))
        return hash_string
    
    @staticmethod
    def unhash_state(h_state:str) -> List[int]:
        state = list(map(int, h_state.split('-')))
        return state
    
    def add_node(self, state:List[int]):
        h = self.hash_state(state)
        if not h in self.nodes.keys():
            self.nodes[h] = Node()

    def has_children(self, state:List[int]) -> bool:
        h = self.hash_state(state)
        return bool(self.nodes[h].actions)
    
    def get_children(self, state:List[int]) -> List[List[int]]:
        h = self.hash_state(state)
        node = self.nodes[h]
        return list(map(self.unhash_state, node.actions.keys()))

    def add_children_to_node(self, state:List[int], children_states:List[List[int]]) -> None:
        h = self.hash_state(state)
        node = self.nodes[h] # Create node if not exists
        if not node.actions:
            # Add children nodes to the tree
            h_children = list(map(self.hash_state, children_states))            
            self.nodes.update({h_child : Node() for h_child in h_children if not h_child in self.nodes.keys()})
            
            # Add actions and children to current state 
            node.actions = {h_child : Action() for h_child in h_children}

    def update_value_visit_action(self, stateA:List[int], stateB:List[int], reward:float) -> None:
        hA = self.hash_state(stateA)
        hB = self.hash_state(stateB)

        node = self.nodes[hA]
        action = node.actions[hB]

        node.increment_visit()
        action.increment_visit()
        action.update_value(reward)

    def reset_search_path(self) -> None:
        self.current_search_path = []
    
    def get_action(self, stateA:List[int], stateB:List[int]) -> Action:
        hA = self.hash_state(stateA)

        if hA in self.nodes.keys():
            hB = self.hash_state(stateB)
            if hB in self.nodes[hA].actions.keys():
                return self.nodes[hA].actions[hB]
        return None
    
    def get_actions(self, state:List[int]) -> List[Action]:
        h = self.hash_state(state)

        if h in self.nodes.keys():
            return list(self.nodes[h].actions.values())
        return None
    
    def get_node(self, state:List[List[int]]) -> Node:
        h = self.hash_state(state)
        if h in self.nodes.keys():
            return self.nodes[h]
        return None
    
    def UCB(self, stateA:List[List[int]], stateB:List[List[int]], C = 3.0e-2) -> float:
        hA = self.hash_state(stateA)
        hB = self.hash_state(stateB)

        node = self.nodes[hA]
        action = node.actions[hB]

        if action.visit == 0:
            return float("+inf")

        return action.value / action.visit + C * np.sqrt(np.log(node.visit) / action.visit)
    
def timing(method_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            self.timings[method_name] += (end_time - start_time)
            return result
        return wrapper
    return decorator

class MCTS():
    PRINT_INFO_STEP = 1
    def __init__(self,
                 simulation_steps:int=10,
                 C:float=np.sqrt(2),
                 **kwargs) -> None:
        """
        MCTS algorithm.
        
        Args:
            - simulation_steps (int)    : Number of simulation steps
            - C (float)                 : Exploration vs Exploitation coefficient in UCB
            - max_depth_selection       : Stop selection phase after <max_depth_selection> steps (to avoid infinite loop)
            - max_solution_search       : Stop search when <max_solution_search> have been found
            - print_info (bool)         : Print current number of nodes in tree and reward
        """
        
        self.simulation_steps = simulation_steps
        self.C = C

        self.tree = Tree()
        self.solution_found = 0

        optional_args = {
            "max_depth_selection" : 10,
            "max_solution_search" : 10,
            "print_info" : False,
        }

        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        
        self.timings = {
            "selection": 0,
            "expansion": 0,
            "simulation": 0,
            "back_propagation": 0
        }

    def is_terminal(self, start_state, state_goal) -> bool:
        return start_state == state_goal
    
    def get_children(self, state):
        """
        Returns the children of a state.
        To override.
        """
        return [state]
    
    def heuristic(self, states, state_goal):
        """
        Default heuristic. Select a node randomly from a set of states.
        To override.
        """
        return np.random.choice(states)
    
    def reward(self, state, state_goal):
        """
        Default reward. Computes the reward associated to the current state.
        To override.
        """
        return np.random.rand()
    
    @timing("selection")
    def selection(self, state, state_goal):
        self.tree.current_search_path = []

        depth = 0
        while depth < self.max_depth_selection:
            self.tree.current_search_path.append(state)

            # Select node that haven't been expanded
            if not self.tree.has_children(state):
                break

            # Select one of the children that haven't been expanded if exists
            children = self.tree.get_children(state)
            unexplored = list(filter(lambda state: not self.tree.has_children(state), children))
            if unexplored:
                state = self.heuristic(unexplored, state_goal)
                self.tree.current_search_path.append(state)
                break

            # Go one level deeper in the tree
            depth += 1
            # If all node have been expanded, select action with maximum UCB score
            state = max(children, key=lambda child_state: self.tree.UCB(state, child_state, self.C))
        
        return state
    
    @timing("expansion")
    def expansion(self, state) -> None:
        # If has no children already
        if not self.tree.has_children(state):
            children_states = self.get_children(state)
            self.tree.add_children_to_node(state, children_states)
        
    @timing("simulation")
    def simulation(self, state, goal_state) -> float:
        terminal_state = False
        for _ in range(self.simulation_steps):

            # Choose successively one child until goal is reached
            if self.tree.has_children(state) and not self.is_terminal(state, goal_state):
                children = self.tree.get_children(state)
                state = self.heuristic(children, goal_state)
            else:
                break

        if self.is_terminal(state, goal_state):
            terminal_state = True

        reward = self.reward(state, goal_state)
        return reward, terminal_state
    
    @timing("back_propagation")
    def back_propagation(self, reward) -> None:
        child_state = self.tree.current_search_path[-1]
        for state in reversed(self.tree.current_search_path[:-1]):
            self.tree.update_value_visit_action(state, child_state, reward)
            child_state = state
    
    def search(self, state_start, state_goal, num_iterations:int=1000):
        
        progress_bar = tqdm.trange(0, num_iterations)
        self.solution_found = 0

        self.tree.reset_search_path()

        for it in progress_bar:
            # Selection
            leaf = self.selection(state_start, state_goal)
            # Expansion
            self.expansion(leaf)
            # Simulation
            reward, terminal_state = self.simulation(leaf, state_goal)
            # Backpropagation
            self.back_propagation(reward)

            if terminal_state:
                self.solution_found += 1

            if self.print_info and it % MCTS.PRINT_INFO_STEP == 0:
                progress_bar.set_postfix({
                        "found": self.solution_found,
                        "nodes": len(self.tree.nodes),
                        "reward": reward})
                
            if self.solution_found >= self.max_solution_search:
                break
                

    def get_best_children(self, state_start, state_goal, n_children:int=1, mode:str="visit") -> List:
    
        def value_child(child_state):
            """
            Function to be maximised by the action
            """
            print("Warning. Value function not set.")
            return 0.

        if mode == "visit":
            def value_child(state, child_state):
                """ Maximum visit
                """
                return self.tree.get_action(state, child_state).visit

        elif mode == "value":
            def value_child(state, child_state):
                """ Maximum average value
                """
                action = self.tree.get_action(state, child_state)
                if action.visit > 0:
                    return action.value / action.visit
                else:
                    return float("-inf")
        
        children = []
        state = state_start
        for _ in range(n_children):

            best_child = max(self.tree.get_node(state).children, key=lambda child : value_child(state, child))
            children.append(best_child)
            state = best_child

            if self.is_terminal(state, state_goal):
                break

        return children
    
    def get_timings(self):
        return self.timings