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

    def compute_eu(self, choice_set, env, user: User):

        if len(choice_set) == 0:
            return -10000

        # Compute first term
        intervention_costs = []
        for action, value, intervention, choice_env, initial_env in choice_set:
            int_cost_tmp = (user.compute_intervention_cost(env, choice_env.copy(), intervention),
                            intervention,
                            choice_env)
            intervention_costs.append(
                (
                    user.compute_choice_probability((action, value, env, choice_env.copy()), env, choice_set),
                    int_cost_tmp
                )
            )

        intervention_costs = [i[0]*i[1][0] for i in intervention_costs]
        return np.sum(intervention_costs)

    def compute_eus(self, env, user, sampled_w, choice_set):

        eus = 0

        if len(sampled_w) == 0:
            return -10000

        previous_weights = env.weights.copy()

        for w in sampled_w:
            env.weights = w
            eus += -self.compute_eu(choice_set, env, user)
        
        env.weights = previous_weights.copy()

        return eus/len(sampled_w)