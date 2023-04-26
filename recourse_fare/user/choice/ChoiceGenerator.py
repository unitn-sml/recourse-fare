import copy
from copy import deepcopy
import numpy as np

from ..user import User

class ChoiceGenerator:
    def __init__(self, k=3):
        self.k = k # Max size of the choice set

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        return result

    def compute_eu(self, choice_set, env, user: User, custom_weights: dict):

        if len(choice_set) == 0:
            return -10000

        # Compute first term
        intervention_eu = []

        # Compute best action given the current environment
        (best_action, best_value, best_intervention, best_previous_state, best_initial_state), intervention_costs = user.compute_best_action(env, choice_set, custom_weights)
        custom_best_action = (best_action, best_value, best_intervention, best_previous_state, best_initial_state)

        for k, (action, value, intervention, choice_env, _) in enumerate(choice_set):

            choice_probability = user.compute_choice_probability((action, value, env, choice_env.copy()),
                env, choice_set, custom_weights=custom_weights, custom_best_action=custom_best_action)

            intervention_eu.append(
                (
                    choice_probability,
                    (intervention_costs[k], intervention, choice_env)
                )
            )

        intervention_eu = [i[0]*i[1][0] for i in intervention_eu]
        return np.sum(intervention_eu)

    def compute_eus(self, env, user, sampled_w, choice_set):

        eus = 0

        if len(sampled_w) == 0:
            return -10000

        eus = sum([self.compute_eu(choice_set, env, user, w) for w in sampled_w])

        return eus/len(sampled_w)