from recourse_fare.environment_w import EnvironmentWeights

from .adult_scm import AdultSCM

from collections import OrderedDict
from typing import Any

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdultEnvironment(EnvironmentWeights):
    """ Simple class showcasing how to build an environment to play with FARE
    """

    def __init__(self, features: dict, weights: dict, model: Any, preprocessor: Any):

        # Preprocessor element. Please have a look below to understand how it is called.
        self.preprocessor = preprocessor

        # The maximum length of an intervention. It considers also the STOP action.
        self.max_intervention_depth = 10

        # Dictionary specifying, for each action, the corresponding implementation of
        # such action in the environment.
        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                'CHANGE_WORKCLASS': self._change_workclass,
                                                'CHANGE_EDUCATION': self._change_education,
                                                'CHANGE_OCCUPATION': self._change_occupation,
                                                'CHANGE_HOURS': self._change_hours,
                                                'CHANGE_CAP_GAIN': self._change_capital_gain,
                                                'CHANGE_CAP_LOSS': self._change_capital_loss
                                                }.items()))
        
        # Dictionary specifying, for each action, the corresponding precondition which needs 
        # to be verified to be able to apply such action. For example, if I already have a bachelor's
        # degree, the action "change_education(high-school diploma)" would be meaningless.
        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._placeholder_stop,
                                                        'CHANGE_WORKCLASS': self._change_workclass_p,
                                                        'CHANGE_EDUCATION': self._change_education_p,
                                                        'CHANGE_OCCUPATION': self._change_occupation_p,
                                                        'CHANGE_HOURS': self._change_hours_p,
                                                        'CHANGE_CAP_GAIN': self._change_capital_gain_p,
                                                        'CHANGE_CAP_LOSS': self._change_capital_loss_p,
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
                                                    'CHANGE_CAP_GAIN': {'index': 5, 'level': 0, 'args': 'CAP'},
                                                    'CHANGE_CAP_LOSS': {'index': 6, 'level': 0, 'args': 'CAP'},
                                                    'INTERVENE': {'index': 7, 'level': 1, 'args': 'NONE'}}.items()))

        # The available arguments. For each type, we need to specify a list of potential values. Each action will be
        # tied to the correspoding type. 
        # The arguments need to contain the NONE type, with a single value 0.
        self.arguments = OrderedDict(sorted({
                                                "WORK": ["Never-worked", "Without-pay", "Self-emp-not-inc", "Self-emp-inc","Private", "Local-gov", "State-gov", "Federal-gov"],
                                                "EDU": ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate', 'Assoc-acdm', 'Assoc-voc', 'Prof-school'],
                                                "OCC": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
                                                "HOUR": list(range(1,25))+list(range(-1,-25)),
                                                "CAP": np.linspace(10,30000, num=50).tolist()+np.linspace(-30000, -10, num=50).tolist(),
                                                "NONE": [0]
                                            }.items()))

        self.program_feature_mapping = {
            'CHANGE_WORKCLASS': "workclass",
            'CHANGE_EDUCATION': "education",
            'CHANGE_OCCUPATION': "occupation",
            'CHANGE_HOURS': "hours_per_week",
            'CHANGE_CAP_GAIN': "capital_gain",
            'CHANGE_CAP_LOSS': "capital_loss"
        }

        scm = AdultSCM(preprocessor)

        # Call parent constructor
        super().__init__(features, weights, scm, model, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_intervention_depth, prog_to_cost=None,
                         program_feature_mapping=self.program_feature_mapping
                         )

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
    
    def _change_capital_gain(self, arguments=None):
        self.features["capital_gain"] += arguments

    def _change_capital_loss(self, arguments=None):
        self.features["capital_loss"] += arguments

    ### ACTIONA PRECONDTIONS

    def _placeholder_stop(self, args=None):
        # We can always call the STOP action.
        return True

    def _change_workclass_p(self, arguments=None):
        return self.arguments["WORK"].index(arguments) != self.arguments["WORK"].index(self.features.get("workclass"))

    def _change_education_p(self, arguments=None):
        return self.arguments["EDU"].index(arguments) != self.arguments["EDU"].index(self.features.get("education"))

    def _change_occupation_p(self, arguments=None):
        return self.arguments["OCC"].index(arguments) != self.arguments["OCC"].index(self.features.get("occupation"))

    def _change_hours_p(self, arguments=None):
        return self.features.get("hours_per_week")+arguments >= 0
    
    def _change_capital_gain_p(self, arguments=None):
        return self.features["capital_gain"]+arguments >= 0

    def _change_capital_loss_p(self, arguments=None):
        return self.features["capital_loss"]+arguments >= 0

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
        obs = self.preprocessor.transform_dict(self.features)
        return torch.FloatTensor(obs)