from rl_mcts.core.environment import Environment
from collections import OrderedDict

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class MockEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim=20):
        super(MockEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, encoding_dim)
        self.l2 = nn.Linear(encoding_dim, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x

class MockEnv(Environment):

    def __init__(self, f,w, config_args=None):

        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                'ADD': self._add,
                                                'SUB': self._sub}.items()))

        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._stop_precondition,
                                                        'ADD': self._add_precondition,
                                                        'SUB': self._sub_precondition,
                                                        'COUNT_10': self._count_10_precondition}.items()))

        self.prog_to_postcondition = OrderedDict(sorted({'COUNT_10': self._count_10_postcondition}.items()))

        self.programs_library = OrderedDict(sorted({'STOP': {'level': -1, 'args': 'NONE'},
                                               'ADD': {'level': 0, 'args': 'INT'},
                                               'SUB': {'level': 0, 'args': 'INT'},
                                               'COUNT_10': {'level': 1, 'args': 'NONE'}}.items()))

        self.arguments = OrderedDict(sorted({
            "INT": list(range(1,6)),
            "NONE": [0]
        }.items()))

        self.complete_arguments = []

        for k, v in self.arguments.items():
            self.complete_arguments += v

        self.arguments_index = [(i, v) for i, v in enumerate(self.complete_arguments)]

        self.max_intervention_depth = {1: 10}

        for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = idx

        # Placeholder needed for testing
        self.data = list(range(0,100))

        super().__init__(f, w, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_intervention_depth, complete_arguments=self.complete_arguments)


    def init_env(self):
        pass

    def reset_env(self, task_index):
        self.has_been_reset = True

        return 0, 0

    def reset_to_state(self, state):
        self.features = state

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    def _stop(self, arguments=None):
        return True

    def _add(self, arguments=None):
        self.features[f"x{arguments}"] += 1

    def _sub(self, arguments=None):
        self.features[f"x{arguments}"] -= 1

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
        return np.sum([self.features.get(k) for k in self.features.keys()]) > 7

    def get_observation(self):
        current_val = np.sum([self.features.get(k) for k in self.features.keys()])
        return torch.FloatTensor(np.array([
            current_val > 0,
            current_val > -0.5,
            current_val > -1,
            current_val > -2,
            current_val > -4,
            current_val > -5,
            current_val > -6,
        ]))

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