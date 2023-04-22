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
        self.has_been_reset = True
        self.act(program, argument)

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


