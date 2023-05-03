from .environment import Environment
from .utils.wfare_utils.structural_weights import StructuralWeights

from typing import Any

import numpy as np
import pandas as pd

class EnvironmentWeights(Environment):
    """Environment which assumes to have user weights."""

    def __init__(self, features, weights, structural_weights, model, prog_to_func, prog_to_precondition,
                 prog_to_postcondition,
                 programs_library, arguments, max_intervention_depth,
                 prog_to_cost,
                 custom_tensorboard_metrics=None,
                 program_feature_mapping: dict=None,
                 ):
        
        self.program_feature_mapping = program_feature_mapping
        assert self.program_feature_mapping, "Mapping between programs and features is missing."

        super().__init__(features, model, prog_to_func, prog_to_precondition,
                         prog_to_postcondition,
                         programs_library, arguments,
                         max_intervention_depth, prog_to_cost,
                         custom_tensorboard_metrics)
        
        self.weights = weights
        self.structural_weights: StructuralWeights = structural_weights
    
    def get_feature_name(self, program_name: str, arguments) -> str:
        return self.program_feature_mapping.get(program_name, None)

    def get_cost(self, program_index: int, args_index: int):
        
        # Get a copy of the previous state
        prev_state = self.features.copy()

        # Get action name and arguments
        program = self.idx_to_prog.get(program_index)
        argument = self.complete_arguments.get(args_index)

        # Compute the cost only if the action is different than the cost
        if program == "STOP":
            return 1
        
        # Get the feature we are going to change
        feature_name = self.get_feature_name(program, argument)

        # Perform the action on the environment
        # We avoid using act() because of a dangerous recoursion loop.
        self.has_been_reset = True
        assert program in self.primary_actions, 'action {} is not defined'.format(program)
        self.prog_to_func[program](argument)

        # Get the new value of the feature
        resulting_value = self.features.get(feature_name)

        # Compute the cost given the structural equations
        action_cost = self.structural_weights.compute_cost(
            feature_name, resulting_value, prev_state.copy(), self.weights
        )

        # Reset the previous environment
        self.features = prev_state

        return action_cost
    
    def get_cost_raw(self, program_index: int, args_value: Any):

        # Get a copy of the previous state
        prev_state = self.features.copy()

        # Get action name and arguments
        program = self.idx_to_prog.get(program_index)

        # Compute the cost only if the action is different than the cost
        if program == "STOP":
            return 1
        
        # Get the feature we are going to change
        feature_name = self.get_feature_name(program, None)

        # Perform the action on the environment
        self.has_been_reset = True
        self.act(program, args_value)

        # Get the new value of the feature
        resulting_value = self.features.get(feature_name)

        # Compute the cost given the structural equations
        action_cost = self.structural_weights.compute_cost(
            feature_name, resulting_value, prev_state.copy(), self.weights
        )

        # Reset the previous environment
        self.features = prev_state

        return action_cost
    
    def get_list_of_costs(self):

        lists_of_costs = []

        # For each available program and argument, compute the cost
        for program in self.programs_library:

            if program == "INTERVENE":
                continue

            available_args = self.arguments.get(self.programs_library.get(program).get("args"))
            
            current_average_cost = []

            for argument in available_args:

                prog_idx = self.prog_to_idx.get(program)
                args_idx = self.inverse_complete_arguments.get(argument)

                if self.can_be_called(prog_idx, args_idx):
                    current_average_cost.append(self.get_cost(prog_idx, args_idx))

            lists_of_costs.append(np.mean(current_average_cost) if len(current_average_cost) > 0 else -1)
        
        # Standardize the costs
        lists_of_costs = np.array(lists_of_costs)
        mask_not_available = np.where(lists_of_costs >= 0, 1, 0)
        
        lists_of_costs = -lists_of_costs
        max_costs = lists_of_costs.max()

        lists_of_costs = lists_of_costs - max_costs
        lists_of_costs = np.exp(lists_of_costs)
        lists_of_costs = lists_of_costs / lists_of_costs.sum()  

        # Negative values are set to -1
        lists_of_costs = mask_not_available*lists_of_costs

        return lists_of_costs


