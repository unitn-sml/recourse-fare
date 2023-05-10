from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
import torch


class Environment():

    def __init__(self, features, model, prog_to_func, prog_to_precondition, prog_to_postcondition, programs_library, arguments,
                 max_intervention_depth, prog_to_cost=None, custom_tensorboard_metrics=None):

        self.features = features
        self.model = model

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

        self.max_intervention_depth = max_intervention_depth

        self.tasks_dict = {}
        self.tasks_list = []

        self.arguments = arguments
        complete_arguments_tmp = [(k, val) for k, v in self.arguments.items() for val in v]
        self.complete_arguments = {idx: v for idx, (k, v) in enumerate(complete_arguments_tmp)}

        self.inverse_complete_arguments = {}
        
        for idx, (k, v) in enumerate(complete_arguments_tmp):
            for p, p_info in self.programs_library.items():
                if v not in self.inverse_complete_arguments:
                    self.inverse_complete_arguments[v] = {}
                if p_info["args"] == k:
                    self.inverse_complete_arguments[v].update({p: idx})

        if custom_tensorboard_metrics is None:
            custom_tensorboard_metrics = {}
        self.custom_tensorboard_metrics = custom_tensorboard_metrics

        self.init_env()

    @abstractmethod
    def get_observation(self):
        pass

    def get_state(self):
        return self.features.copy()

    def reset_env(self):
        self.has_been_reset = True
        return 0, 0

    def init_env(self):
        self.has_been_reset = True

    def get_obs_dimension(self):
        return len(self.get_observation())

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
        return self.max_intervention_depth

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

        mask = []
        for k, r in self.arguments.items():
            if k == permitted_arguments:
                mask.append(np.ones(len(r)))
            else:
                mask.append(np.zeros(len(r)))

        return np.concatenate(mask, axis=None)

    def can_be_called(self, program_index, args_index):

        if args_index == None:
            return False

        program = self.get_program_from_index(program_index)
        args = self.complete_arguments.get(args_index)

        mask_over_args = self.get_mask_over_args(program_index)
        if mask_over_args[args_index] == 0:
            return False

        return self.prog_to_precondition[program](args)

    def get_cost(self, program_index, args_index):

        if self.prog_to_cost is None:
            return 0

        program = self.get_program_from_index(program_index)
        args = self.complete_arguments.get(args_index)

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

    def get_additional_parameters(self):
        return {
            "argument_types": self.arguments
        }

    def get_state_str(self, state):
        return ""

    def compare_state(self, state_a, state_b):
        return state_a == state_b

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]
    
    def get_current_actions(self):

        current_actions = []
        for precondition,f in self.prog_to_precondition.items():
            if self.programs_library.get(precondition).get("level") <= 0 and precondition != "STOP":
                function_args = self.arguments.get(self.programs_library.get(precondition).get("args"))
                for arg in function_args:
                    prog_idx = self.prog_to_idx.get(precondition, None)
                    args_idx = self.inverse_complete_arguments.get(arg).get(precondition)
                    assert prog_idx is not None and args_idx is not None
                    if self.can_be_called(prog_idx, args_idx):
                        current_actions.append([precondition, arg, prog_idx, args_idx])

        return current_actions