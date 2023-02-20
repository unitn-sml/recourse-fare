from abc import ABC

class ExecutionTrace(ABC):
    """
    Object which contains a full execution trace extracted by MCTS with the agent.
    """

    def __init__(self, lstm_states, lstm_args_states, programs_index, observations, previous_actions, task_reward, program_arguments,
                 rewards, mcts_policies,clean_sub_execution = True):

        self.lstm_states = lstm_states
        self.programs_index = programs_index
        self.observations = observations
        self.previous_actions = previous_actions
        self.task_reward = task_reward
        self.program_arguments = program_arguments
        self.rewards = rewards
        self.mcts_policies = mcts_policies
        self.clean_sub_execution = clean_sub_execution 
        self.lstm_states_args = lstm_args_states

    def get_trace_programs(self):
        result =  [(p, arg) for p, arg in zip(self.previous_actions, self.program_arguments)]
        # Discard the first element (since it will have a None action)
        return result[1:]

    def flatten(self):
        return list(zip(self.observations,
                        self.programs_index,
                        self.lstm_states,
                        self.mcts_policies,
                        self.rewards,
                        self.program_arguments,
                        self.lstm_states_args))