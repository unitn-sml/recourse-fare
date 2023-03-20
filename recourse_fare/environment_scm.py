from .environment import Environment

from causalgraphicalmodels import StructuralCausalModel

import numpy as np
import pandas as pd

class EnvironmentSCM(Environment):
    """Environment which uses an SCM to compute the results."""

    def __init__(self, features, model, prog_to_func, prog_to_precondition,
                 prog_to_postcondition,
                 programs_library, arguments, max_intervention_depth, prog_to_cost=None,
                 custom_tensorboard_metrics=None,
                 program_feature_mapping: dict= None,
                 scm: StructuralCausalModel=None, A:np.array=None):

        super().__init__(features, model, prog_to_func, prog_to_precondition,
                         prog_to_postcondition,
                         programs_library, arguments,
                         max_intervention_depth, prog_to_cost,
                         custom_tensorboard_metrics)
        
        self.program_feature_mapping = program_feature_mapping
        assert self.program_feature_mapping, "Mapping between programs and features is missing."
        
        # Sampling budget from the SCM to compute the expected result
        self.scm_samples = 50

        # SCM structure (this is a placehoder)
        self.scm = scm

        # SCM cost matrix
        # A = {
        # x1: [x11, x12, x13],
        # x2: [x21, x22, x23],
        # x3: [x31, x32, x31]
        # }
        self.A = A

    def get_feature_name(self, program_name: str, arguments) -> str:
        return self.program_feature_mapping.get(program_name, None)
    
    def act(self, primary_action, arguments=None):
        
        assert self.has_been_reset, 'Need to reset the environment before acting'
        assert primary_action in self.primary_actions, 'action {} is not defined'.format(primary_action)
        
        # Given the action, we return the feature its modifies
        feature_name = self.get_feature_name(primary_action, arguments)

        # We apply the action and we get the result for the given feature
        self.prog_to_func[primary_action](arguments)
        new_value =self.features.get(feature_name)

        # Perform the new action and sample a new results
        scm_do = self.scm.do(feature_name)
        new_env = scm_do.sample(n_samples=self.scm_samples,
                      set_values={feature_name: [new_value for _ in range(self.scm_samples)]})
        
        # We compute the average of the changes (expectation)
        new_env = new_env.mean()

        # We regenerate the environment based on the SCM
        self.features = {k:new_env[k] for k in new_env.index}

        # Return an observation
        return self.get_observation()
    
    def get_cost(self, program_index: int, args_index: int):
        
        # Get a copy of the previous state
        prev_state = self.features.copy()

        # Get action name and arguments
        program = self.idx_to_prog.get(program_index)
        argument = self.complete_arguments[args_index]

        # Compute the cost only if the action is different than the cost
        if program == "STOP":
            return 1
        
        # Get the feature we are going to change
        feature_name = self.get_feature_name(program, argument)

        # Perform the action on the environment
        self.has_been_reset = True
        self.act(program, argument)

        # Compute the causal cost
        causal_cost = np.sum([self.A.get(feature_name).get(k, 0)*np.abs(self.features[k]-prev_state[k]) for k in prev_state.keys()])

        # Reset the previous environment
        self.features = prev_state

        return causal_cost


