from abc import ABC

class MCTSNode(ABC):

    def __init__(self, *initial_data, **kwargs):
        """
        https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        :param initial_data:
        :param kwargs:
        """

        self.parent = None
        self.childs = []
        self.visit_count = 0
        self.total_action_value = []
        self.prior = None
        self.program_index = None
        self.program_from_parent_index = None
        self.observation = None
        self.env_state = None
        self.h_lstm = None
        self.c_lstm = None
        self.h_lstm_args = None
        self.c_lstm_args = None
        self.depth = 0
        self.selected = False
        self.args = None
        self.args_index = None
        self.denom = 0.0
        self.estimated_qval = 0.0
        self.cost = 0
        self.single_action_cost = 0

        self.program_name = None

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @staticmethod
    def initialize_root_args(task_index, init_observation, env_state, h, c, h_args, c_args):
        return MCTSNode({
            "parent": None,
            "childs": [],
            "visit_count": 1,
            "total_action_value": [],
            "prior": None,
            "program_index": task_index,
            "program_from_parent_index": None,
            "observation": init_observation.clone(),
            "env_state": env_state.copy(),
            "h_lstm": h.clone(),
            "c_lstm": c.clone(),
            "h_lstm_args": h_args.clone(),
            "c_lstm_args": c_args.clone(),
            "depth": 0,
            "selected": True,
            "args": 0,
            "args_index": None,
            "denom": 0.0,
            "estimated_qval": 0.0,
            "program_name": None,
            "cost": 0.0,
            "single_action_cost": 0.0
        })

    def to_dict(self):
        return {
            "parent": self.parent,
            "childs": self.childs,
            "visit_count": self.visit_count,
            "total_action_value": self.total_action_value,
            "prior": self.prior,
            "program_index": self.program_index,
            "program_from_parent_index": self.program_from_parent_index,
            "observation": self.observation,
            "env_state": self.env_state,
            "h_lstm": self.h_lstm,
            "c_lstm": self.c_lstm,
            "depth": self.depth,
            "selected": self.selected,
            "args": self.args,
            "args_index": self.args_index,
            "denom": self.denom,
            "estimated_qval": self.estimated_qval,
            "h_lstm_args": self.h_lstm_args.clone(),
            "c_lstm_args": self.c_lstm_args.clone(),
            "program_name": self.program_name,
            "cost": self.cost,
            "single_action_cost": self.single_action_cost
        }