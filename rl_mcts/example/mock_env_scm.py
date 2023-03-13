from rl_mcts.core.environment_scm import EnvironmentSCM
from collections import OrderedDict

from causalgraphicalmodels import StructuralCausalModel

import numpy as np
import torch

class MockEnv(EnvironmentSCM):

    def __init__(self, f, config_args=None):

        self.program_feature_mapping = {
            "ADD": lambda x: f"x{x}",
            "SUB": lambda x: f"x{x}"
        }

        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                'ADD': self._add,
                                                'SUB': self._sub}.items()))

        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._stop_precondition,
                                                        'ADD': self._add_precondition,
                                                        'SUB': self._sub_precondition}.items()))

        self.prog_to_postcondition = self._count_10_postcondition

        self.programs_library = OrderedDict(sorted({'STOP': {'index': 0, 'level': -1, 'args': 'NONE'},
                                               'ADD': {'index': 1, 'level': 0, 'args': 'INT'},
                                               'SUB': {'index': 2, 'level': 0, 'args': 'INT'},
                                               'COUNT_10': {'index': 3, 'level': 1, 'args': 'NONE'}}.items()))

        self.arguments = OrderedDict(sorted({
            "INT": list(range(1,6)),
            "NONE": [0]
        }.items()))

        self.complete_arguments = []

        for k, v in self.arguments.items():
            self.complete_arguments += v

        self.max_depth_dict = 5

        scm = StructuralCausalModel({
                "x1": lambda n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
                "x2": lambda x1, n_samples: np.random.normal(loc=x1*2, scale=0.1),
                "x3": lambda x1, x2, n_samples: x2 ** 2 - x1,
                "x4": lambda x1, n_samples: np.random.normal(loc=x1, scale=0.3),
                "x5": lambda x2, x3, x4, n_samples: np.random.normal(loc=x4, scale=0.1) + x2 -x3,
            })
        
        A = {
            "x1": {},
            "x2": {"x1": 0.2},
            "x3": {"x1": 0.2, "x2": 1.2},
            "x4": {"x1": 0.2},
            "x5": {"x2": 2, "x3": 1.2, "x4": -0.3},
        }

        super().__init__(f, None, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_depth_dict,
                         complete_arguments=self.complete_arguments,
                         scm=scm,
                         A=A,
                         program_feature_mapping=self.program_feature_mapping)
        
    def get_feature_name(self, program_name: str, arguments) -> str:
        return self.program_feature_mapping.get(program_name, None)(arguments)

    def init_env(self):
        self.has_been_reset = True

    def reset_env(self):
        self.has_been_reset = True

        return 0, 0

    def reset_to_state(self, state):
        self.features = state

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    def _stop(self, arguments=None):
        return True

    def _add(self, arguments=None):
        self.features[f"x{arguments}"] += 0.5

    def _sub(self, arguments=None):
        self.features[f"x{arguments}"] -= 0.5

    def _stop_precondition(self, args):
        return True

    def _add_precondition(self, args):
        return True

    def _sub_precondition(self, args):
        return self.features.get(f"x{args}") > 0

    def _count_10_precondition(self, args):
        return True

    def _count_10_postcondition(self, init_state, current_state):
        # TODO: testing only!!!! Change this!!!! It will return always true to facilitate testing.
        #return True
        prev_sum = np.sum([init_state.get(k) for k in init_state.keys()])
        return np.sum([self.features.get(k) for k in self.features.keys()]) > 7 and prev_sum < 7 

    def get_observation(self):
        current_val = [self.features.get(k) for k in self.features.keys()]
        return torch.FloatTensor(np.array(current_val))

    def get_state(self):
        return self.features.copy()

    def get_obs_dimension(self):
        return len(self.get_observation())

    def compare_state(self, state_a, state_b):
        return state_a == state_b

    def get_mask_over_args(self, program_index):
        """
        Return the available arguments which can be called by that given program
        :param program_index: the program index
        :return: a max over the available arguments
        """

        program = self.get_program_from_index(program_index)
        permitted_arguments = self.programs_library[program]["args"]

        mask = []
        for k, r in self.arguments.items():
            if k == permitted_arguments:
                mask.append(np.ones(len(r)))
            else:
                mask.append(np.zeros(len(r)))

        return np.concatenate(mask, axis=None)

    def get_additional_parameters(self):
        return {
            "types": self.arguments
        }

if __name__ == "__main__":

    env = MockEnv()

    env.init_env()
    print(env.memory)
    print(env._count_10_postcondition(None, None))
    env._add()
    print(env.memory)
    print(env._count_10_postcondition(None, None))