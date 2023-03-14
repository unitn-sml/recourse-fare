from rl_mcts.core.environment_scm import EnvironmentSCM
from collections import OrderedDict

from causalgraphicalmodels import StructuralCausalModel

import numpy as np
import pandas as pd
import torch

class MockEnv(EnvironmentSCM):

    def __init__(self, f, model, preprocessing):

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

        self.max_depth_dict = 5

        self.preprocessing = preprocessing
        self.feature_ordering_prep = list(self.preprocessing.feature_names_in_)

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

        super().__init__(f, model, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_depth_dict,
                         scm=scm,
                         A=A,
                         program_feature_mapping=self.program_feature_mapping)
        
    def get_feature_name(self, program_name: str, arguments) -> str:
        return self.program_feature_mapping.get(program_name, None)(arguments)

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
        return self.model(self.features) > 7

    def get_observation(self):
        return torch.FloatTensor(
            self.preprocessing.transform(
            pd.DataFrame.from_records([{k:self.features.get(k) for k in self.feature_ordering_prep}])
        ))

    def get_state(self):
        return self.features.copy()
    
    def reset_to_state(self, state):
        self.features = state

    def get_additional_parameters(self):
        return {
            "types": self.arguments
        }