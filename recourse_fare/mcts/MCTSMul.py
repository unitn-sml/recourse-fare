from ..mcts.MCTSWeights import MCTSWeights
from ..mcts.MCTSNode import MCTSNode

import torch
import numpy as np

from typing import Union

def mcts_sigmoid(x, eta=0.01):
    return 1/(1+np.exp(-x*eta))

class MCTSMul(MCTSWeights):

    def __init__(self, environment, policy, number_of_simulations: int = 100, exploration=True, dir_noise: float = 0.03, dir_epsilon: float = 0.3, level_closeness_coeff: float = 3, level_0_penalty: float = 1, qvalue_temperature: float = 1, temperature: float = 1.3, c_puct: float = 0.5,
                 gamma: float = 0.97, action_cost_coeff: float = 1, action_duplicate_cost: float = 1, minimum_cost: float = 10000,
                 previous_succesfull_solutions: list = []) -> None:
        
        self.previous_succesfull_solutions = previous_succesfull_solutions
        
        super().__init__(environment, policy, number_of_simulations, exploration, dir_noise, dir_epsilon, level_closeness_coeff, level_0_penalty, qvalue_temperature, temperature, c_puct, gamma, action_cost_coeff, action_duplicate_cost, minimum_cost)

    def _play_episode(self, root_node: MCTSNode, deterministic_actions: list=None):
        stop = False
        max_depth_reached = False
        illegal_action = False

        total_node_expanded_simulation = 0

        while not stop and not max_depth_reached and not illegal_action and self.clean_sub_executions:

            root_node.selected = True

            if root_node.depth >= self.env.get_max_depth():
                max_depth_reached = True

            else:
                env_state = root_node.env_state.copy()

                # record obs, progs and lstm states only if they correspond to the current task at hand
                self.lstm_states.append((root_node.h_lstm, root_node.c_lstm))
                self.lstm_args_states.append((root_node.h_lstm_args, root_node.c_lstm_args))
                self.programs_index.append(root_node.program_index)
                self.observations.append(root_node.observation.clone())
                self.previous_actions.append(root_node.program_from_parent_index)
                self.program_arguments.append(root_node.args)
                self.rewards.append(None)

                # Spend some time expanding the tree from your current root node
                for _ in range(self.number_of_simulations):
                    # run a simulation
                    self.recursive_call = False
                    simulation_max_depth_reached, has_expanded_node, node, value, failed_simulation, node_expanded = self._simulate(
                        root_node, deterministic_actions)

                    total_node_expanded_simulation += node_expanded

                    # get reward
                    if failed_simulation:
                        value = -1.0
                    elif not simulation_max_depth_reached and not has_expanded_node:
                        # if node corresponds to end of an episode, backprogagate real reward
                        reward = self.env.get_reward()

                        # Check if we already reached this state (which means we applied the same actions)
                        # If that it the case, the we place a negative reward
                        if self.env.features.copy() in self.previous_succesfull_solutions:
                            reward = -1 

                        if reward > 0:
                            value = reward * (self.gamma ** node.cost) * (self.gamma ** node.depth) * mcts_sigmoid(self.minimum_cost-node.cost)
                        else:
                            value = -1

                    elif simulation_max_depth_reached:
                        # if episode stops because the max depth allowed was reached, then reward = -1
                        value = -1.0

                    value = float(value)

                    exp_val = torch.exp(self.qvalue_temperature*torch.FloatTensor([value]))

                    # Propagate information backwards
                    while node.parent is not None:
                        node.visit_count += 1
                        node.total_action_value.append(value)

                        node.denom += exp_val
                        softmax = exp_val / node.denom
                        node.estimated_qval += softmax * torch.FloatTensor([value])

                        node = node.parent

                    # Root node is not included in the while loop
                    self.root_node.total_action_value.append(value)
                    self.root_node.visit_count += 1

                    self.root_node.denom += exp_val
                    softmax = exp_val / self.root_node.denom
                    self.root_node.estimated_qval += softmax * torch.FloatTensor([value])

                    # Go back to current env state
                    self.env.reset_to_state(env_state.copy())

                # Sample next action
                mcts_policy, args_policy, program_to_call_index, args_to_call_index = self._sample_policy(root_node)

                 # Force the deterministic action (this is for inference only)
                if deterministic_actions is not None and len(deterministic_actions) > 0 and root_node.depth < len(
                        deterministic_actions):
                    program_to_call_index = deterministic_actions[node.depth][0]
                    args_to_call_index = deterministic_actions[node.depth][1]

                # Set new root node
                root_node = [child for child in root_node.childs
                                 if child.program_from_parent_index == program_to_call_index
                                 and child.args_index == args_to_call_index]

                # If we choose an illegal action from this point, we exit
                if len(root_node) == 0:
                    root_node = None
                    illegal_action = True
                else:
                    root_node = root_node[0]

                # Record mcts policy
                self.mcts_policies.append(torch.cat([mcts_policy, args_policy], dim=1))

                # Apply chosen action
                if not illegal_action:
                    if program_to_call_index == self.env.programs_library["STOP"]['index']:
                        stop = True
                    else:
                        self.env.reset_to_state(root_node.env_state.copy())

        return root_node, max_depth_reached, illegal_action, total_node_expanded_simulation