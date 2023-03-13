from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
import torch


class Environment(ABC):

    def __init__(self, features, weights, prog_to_func, prog_to_precondition, prog_to_postcondition, programs_library, arguments,
                 max_depth_dict, prog_to_cost=None, complete_arguments=None, custom_tensorboard_metrics=None):

        self.weights = weights
        self.features = features

        self.prog_to_func = prog_to_func
        self.prog_to_precondition = prog_to_precondition
        self.prog_to_postcondition = prog_to_postcondition
        self.prog_to_cost = prog_to_cost
        self.programs_library = programs_library

        self.programs = list(self.programs_library.keys())
        self.primary_actions = [prog for prog in self.programs_library if self.programs_library[prog]['level'] <= 0]
        self.mask = dict(
            (p, self._get_available_actions(p)) for p in self.programs_library if self.programs_library[p]["level"] > 0)

        self.prog_to_idx = dict((prog, elems["index"]) for prog, elems in self.programs_library.items())
        self.idx_to_prog = dict((idx, prog) for (prog, idx) in self.prog_to_idx.items())

        self.has_been_reset = True

        self.max_depth_dict = max_depth_dict

        self.tasks_dict = {}
        self.tasks_list = []

        self.arguments = arguments
        self.complete_arguments = complete_arguments

        if custom_tensorboard_metrics is None:
            custom_tensorboard_metrics = {}
        self.custom_tensorboard_metrics = custom_tensorboard_metrics

        self.init_env()

    def setup_system(self, boolean_cols, categorical_cols, encoder, scaler,
                      classifier, net_class, net_layers=5, net_size=108):
        
        self.parsed_columns = boolean_cols + categorical_cols

        self.complete_arguments = []

        for k, v in self.arguments.items():
            self.complete_arguments += v

        self.arguments_index = [(i, v) for i, v in enumerate(self.complete_arguments)]

        self.max_depth_dict = {1: 5}

        for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = idx

        # Load encoder
        self.data_encoder = pickle.load(open(encoder, "rb"))
        self.data_scaler = pickle.load(open(scaler, "rb"))

        # Load the classifier
        if net_class:
            checkpoint = torch.load(classifier)
            self.classifier = net_class(net_size, layers=net_layers)  # Taken empirically from the classifier
            self.classifier.load_state_dict(checkpoint)
        else:
            self.classifier = None

        # Custom metric we want to print at each iteration
        self.custom_tensorboard_metrics = {
            "call_to_the_classifier": 0
        }

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def reset_env(self):
        pass

    @abstractmethod
    def init_env(self):
        pass

    @abstractmethod
    def get_obs_dimension(self):
        pass

    @abstractmethod
    def reset_to_state(self, state) -> None:
        """Reset the state of the environment to the one given as argument.

        Args:
            state: new state which will replace the current one.
        """
        pass

    def get_num_programs(self):
        return len(self.programs)

    def start_task(self):
        
        # Reset the environment and save the task initial state
        self.reset_env()
        self.task_init_state = self.get_state()

        return self.get_observation()

    def end_task(self):
        """
        Ends the last tasks that has been started.
        """
        self.has_been_reset = False

    def get_max_depth(self):
        return self.max_depth_dict

    def _get_available_actions(self, program):
        level_prog = self.programs_library[program]["level"]
        assert level_prog > 0
        mask = np.zeros(len(self.programs))
        for prog, elems in self.programs_library.items():
            if elems["level"] < level_prog:
                mask[elems["index"]] = 1
        return mask

    def get_program_from_index(self, program_index):
        """Returns the program name from its index.
        Args:
          program_index: index of desired program
        Returns:
          the program name corresponding to program index
        """
        return self.idx_to_prog[program_index]

    def get_program_level(self, program):
        return self.programs_library[program]['level']

    def get_program_level_from_index(self, program_index):
        """
        Args:
            program_index: program index
        Returns:
            the level of the program
        """
        program = self.get_program_from_index(program_index)
        return self.programs_library[program]['level']

    def get_mask_over_actions(self, program_index):

        program = self.get_program_from_index(program_index)
        assert program in self.mask, "Error program {} provided is level 0".format(program)
        mask = self.mask[program].copy()
        # remove actions when pre-condition not satisfied
        for program, program_dict in self.programs_library.items():
            if not self.prog_to_precondition[program]():
                mask[program_dict['index']] = 0
        return mask

    def get_mask_over_args(self, program_index):
        """
        Return the available arguments which can be called by that given program
        :param program_index: the program index
        :return: a max over the available arguments
        """

        program = self.get_program_from_index(program_index)
        permitted_arguments = self.programs_library[program]["args"]
        mask = np.zeros(len(self.arguments))
        for i in range(len(self.arguments)):
            if sum(self.arguments[i]) in permitted_arguments:
                mask[i] = 1
        return mask

    def can_be_called(self, program_index, args_index):
        program = self.get_program_from_index(program_index)
        args = self.complete_arguments[args_index]

        mask_over_args = self.get_mask_over_args(program_index)
        if mask_over_args[args_index] == 0:
            return False

        return self.prog_to_precondition[program](args)

    def get_cost(self, program_index, args_index):

        if self.prog_to_cost is None:
            return 0

        program = self.get_program_from_index(program_index)
        args = self.complete_arguments[args_index]

        return self.prog_to_cost[program](args)


    def act(self, primary_action, arguments=None):
        assert self.has_been_reset, 'Need to reset the environment before acting'
        assert primary_action in self.primary_actions, 'action {} is not defined'.format(primary_action)
        self.prog_to_func[primary_action](arguments)
        return self.get_observation()

    def get_reward(self):
        task_init_state = self.task_init_state
        state = self.get_state()
        current_task_postcondition = self.prog_to_postcondition
        return int(current_task_postcondition(task_init_state, state))

    @abstractmethod
    def get_additional_parameters(self):
        return {}

    def get_state_str(self, state):
        return ""

    @abstractmethod
    def compare_state(self, state_a, state_b):
        pass