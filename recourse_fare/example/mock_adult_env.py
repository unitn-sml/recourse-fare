from recourse_fare.environment import Environment

from collections import OrderedDict
from typing import Any

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdultEnvironment(Environment):
    """ Simple class showcasing how to build an environment to play with FARE
    """

    def __init__(self, features: dict, model: Any, preprocessor: Any):
        """Constructor of the class. It takes two mandatory element: the features
        as a dictionary (e.g., {"age": 25, "job": "scientist", ...}) and the black-box
        model which we will use. This class also takes as additional parameter a 
        preprocessor (which needs to implement a transform() method), which can be used
        to preprocess the data before feeding them to the black-box model and the FARE
        model itself.

        Each environment MUST specify the following member variables:
        - self.max_intervention_depth
        - self.prog_to_func
        - self.prog_to_precondition
        - self.prog_to_postcondition
        - self.programs_library
        - self.arguments
        You can find the description in the code below.

        :param features: user features
        :type features: dict
        :param model: black-box model 
        :type model: Any
        :param preprocessor: preprocessor
        :type preprocessor: Any
        """

        # Preprocessor element. Please have a look below to understand how it is called.
        self.preprocessor = preprocessor

        # The maximum length of an intervention. It considers also the STOP action.
        self.max_intervention_depth = 7

        # Dictionary specifying, for each action, the corresponding implementation of
        # such action in the environment.
        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                'CHANGE_WORKCLASS': self._change_workclass,
                                                'CHANGE_EDUCATION': self._change_education,
                                                'CHANGE_OCCUPATION': self._change_occupation,
                                                'CHANGE_HOURS': self._change_hours
                                                }.items()))
        
        # Dictionary specifying, for each action, the corresponding precondition which needs 
        # to be verified to be able to apply such action. For example, if I already have a bachelor's
        # degree, the action "change_education(high-school diploma)" would be meaningless.
        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._placeholder_stop,
                                                        'CHANGE_WORKCLASS': self._change_workclass_p,
                                                        'CHANGE_EDUCATION': self._change_education_p,
                                                        'CHANGE_OCCUPATION': self._change_occupation_p,
                                                        'CHANGE_HOURS': self._change_hours_p,
                                                        }.items()))

        # Function which validate the environment and it checks if we reached recourse.
        self.prog_to_postcondition = self._intervene_postcondition

        # Programs library. It contains all the available actions the user can perform. For each action, we need to
        # specify three things: the index (it goes from 0 to n), the level and the argument type that function accepts.
        #
        # Here we have two small caveats:
        #  * level: 1 represent the program we want to learn (INTERVENE), 0 represents the action we can take and
        # -1 represents the stop action, which is called to signal the termination of an intervention;
        # * The program library MUST contain the STOP and INTERVENE programs as defined below;
        self.programs_library = OrderedDict(sorted({'STOP': {'index': 0, 'level': -1, 'args': 'NONE'},
                                                    'CHANGE_WORKCLASS': {'index': 1, 'level': 0, 'args': 'WORK'},
                                                    'CHANGE_EDUCATION': {'index': 2, 'level': 0, 'args': 'EDU'},
                                                    'CHANGE_OCCUPATION': {'index': 3, 'level': 0, 'args': 'OCC'},
                                                    'CHANGE_HOURS': {'index': 4, 'level': 0, 'args': 'HOUR'},
                                                    'INTERVENE': {'index': 5, 'level': 1, 'args': 'NONE'}}.items()))

        # The available arguments. For each type, we need to specify a list of potential values. Each action will be
        # tied to the correspoding type. 
        # The arguments need to contain the NONE type, with a single value 0.
        self.arguments = OrderedDict(sorted({
                                                "WORK": ["Never-worked", "Without-pay", "Self-emp-not-inc", "Self-emp-inc","Private", "Local-gov", "State-gov", "Federal-gov"],
                                                "EDU": ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate', 'Assoc-acdm', 'Assoc-voc', 'Prof-school'],
                                                "OCC": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
                                                "HOUR": list(range(0,25)),
                                                "NONE": [0]
                                            }.items()))

        # Dictionary which specifies, for each action, the corresponding function that returns the cost of applied
        # said action on the current environment.
        self.prog_to_cost = OrderedDict(sorted({'STOP': self._stop_cost,
                                                'CHANGE_WORKCLASS': self._change_workclass_cost,
                                                'CHANGE_EDUCATION': self._change_education_cost,
                                                'CHANGE_OCCUPATION': self._change_occupation_cost,
                                                'CHANGE_HOURS': self._change_hours_cost
                                                }.items()))

        self.cost_per_argument = {
            "WORK": {"Never-worked": 4, "Without-pay":5, "Self-emp-not-inc":6, "Self-emp-inc": 6,
                     "Private":7, "Local-gov": 7, "State-gov":8, "Federal-gov":8},
            "OCC": {"Tech-support": 8,
                     "Craft-repair": 6,
                     "Other-service": 6,
                     "Sales": 8,
                     "Exec-managerial": 9,
                     "Prof-specialty": 8,
                     "Handlers-cleaners": 7,
                     "Machine-op-inspct":7,
                     "Adm-clerical":8,
                     "Farming-fishing":6,
                     "Transport-moving":6,
                     "Priv-house-serv":6,
                     "Protective-serv":6,
                     "Armed-Forces":6
                    },
            "REL": {"Wife": 5, "Own-child":6, "Husband":5, "Not-in-family":4, "Other-relative":4, "Unmarried":4}

        }

        # Call parent constructor
        super().__init__(features, model, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_intervention_depth,
                         prog_to_cost=self.prog_to_cost)

    # Some utilites functions

    def reset_to_state(self, state):
        self.features = state.copy()

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    ### ACTIONS

    def _stop(self, arguments=None):
        return True

    def _change_workclass(self, arguments=None):
        self.features["workclass"] = arguments

    def _change_education(self, arguments=None):
        self.features["education"] = arguments

    def _change_occupation(self, arguments=None):
        self.features["occupation"] = arguments

    def _change_relationship(self, arguments=None):
        self.features["relationship"] = arguments

    def _change_hours(self, arguments=None):
        self.features["hours_per_week"] += arguments

    ### ACTIONA PRECONDTIONS

    def _placeholder_stop(self, args=None):
        # We can always call the STOP action.
        return True

    def _change_workclass_p(self, arguments=None):
        return self.cost_per_argument["WORK"][arguments] >= self.cost_per_argument["WORK"][self.features.get("workclass")]

    def _change_education_p(self, arguments=None):
        return self.arguments["EDU"].index(arguments) > self.arguments["EDU"].index(self.features.get("education"))

    def _change_occupation_p(self, arguments=None):
        return self.cost_per_argument["OCC"][arguments] >= self.cost_per_argument["OCC"][self.features.get("occupation")]

    def _change_relationship_p(self, arguments=None):

        if arguments == "Wife":
            return self.features.get("relationship") != "Husband"
        elif arguments == "Husband":
            return self.features.get("relationship") != "Wife"

        return self.cost_per_argument["REL"][arguments] >= self.cost_per_argument["REL"][self.features.get("relationship")]

    def _change_hours_p(self, arguments=None):
        return self.features.get("hours") == arguments

    ### COSTS

    def _stop_cost(self, arguments=None):
        return 1

    def _change_workclass_cost(self, arguments=None):
        return (self.cost_per_argument.get("WORK").get(arguments)+10) * self._rescale_by_edu()

    def _change_education_cost(self, arguments=None):
        return self.arguments.get("EDU").index(arguments)+10

    def _change_occupation_cost(self, arguments=None):
        return (self.cost_per_argument.get("OCC").get(arguments)+10) * self._rescale_by_workclass()

    def _change_relationship_cost(self, arguments=None):
        return self.cost_per_argument.get("REL").get(arguments)+10

    def _change_hours_cost(self, arguments=None):
        return self.arguments["HOUR"].index(arguments)+10

    # Rescaling

    def _rescale_by_edu(self):
        return 1/(len(self.arguments["EDU"])-self.arguments.get("EDU").index(self.features.get("education")))

    def _rescale_by_workclass(self):
        return 1/(len(self.arguments["WORK"])-self.arguments.get("WORK").index(self.features.get("workclass")))

    ### POSTCONDITIONS

    def _intervene_postcondition(self, init_state, current_state):
        # We basically check if the model predicts a 0 (which means
        # recourse) given the current features. We discard the 
        # init_state
        obs = self.preprocessor.transform(
            pd.DataFrame.from_records(
                [current_state]
            )
        )
        return self.model.predict(obs)[0] == 0

    ## OBSERVATIONS

    def get_observation(self):
        # We returns a preprocessed version of the features
        # as a torch tensor.

        obs = self.preprocessor.transform(
            pd.DataFrame.from_records(
                [self.features]
            )
        )
        return torch.FloatTensor(obs)