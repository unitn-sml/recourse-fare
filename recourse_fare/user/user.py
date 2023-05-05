from abc import abstractmethod, ABC

from ..environment_w import EnvironmentWeights
from ..utils.wfare_utils.structural_weights import StructuralWeights
from ..utils.functions import compute_intervention_cost

import numpy as np



class User():

    def __init__(self):
        pass

    @abstractmethod
    def compute_best_action(self, current_env: EnvironmentWeights, choice_set: list, custom_weights: dict=None):
        pass

    @abstractmethod
    def compute_choice_probability(self, action, value, current_env: EnvironmentWeights, choice_set: list, custom_weights: dict=None) -> float:
        pass

class NoiselessUser(User):

    def __init__(self):
        super().__init__()

    def compute_best_action(self, env: EnvironmentWeights, choice_set: list, custom_weights: dict=None) -> list:
        """
        Return the best action for this user, given a choice set
        :param choice_set: choice set with the candidate interventions and action/value
        :param estimated: if True, compute the best action using the estimated weights
        :return: the corresponding best action
        """

        if len(choice_set) == 1:
            return choice_set[0], [0]

        intervention_cost_values = []
        for _, _, intervention, current_env, _ in choice_set:
            int_cost_tmp = compute_intervention_cost(env, current_env, intervention, custom_weights)
            intervention_cost_values.append(int_cost_tmp)

        return choice_set[intervention_cost_values.index(min(intervention_cost_values))], intervention_cost_values

    def compute_choice_probability(self, action, current_env: EnvironmentWeights, choice_set: list, custom_weights: dict=None, custom_best_action: list=None) -> float:
        """
        Compute the probability of choosing a given action in a choice set
        :param action: action/value pair we want to get the probability from
        :param choice_set: the choice set from which we need to estimate the probability
        :param estimated: if True, use the estimated weights and not the ground truth
        :return: probability of choosing the given action
        """
        
        if not custom_best_action:
            best_action, best_argument, best_intervention, _, _ = self.compute_best_action(current_env, choice_set, custom_weights)
        else:
            best_action, best_argument, best_intervention, _, _ = custom_best_action
        
        probability_value = 1.0 if action[0] == best_action and action[1] == best_argument and action[2] == best_intervention else 0.0
        return probability_value

class LogisticNoiseUser(User):

    def __init__(self, temperature: float=0.9):
        self.temperature = temperature
        super().__init__()

    def compute_probabilities(self, env, choice_set, custom_weights):

        intervention_cost_values = []
        intervention_cost_values_original = []
        for _, _, intervention, current_env, _ in choice_set:
            int_cost_tmp = compute_intervention_cost(env, current_env, intervention, custom_weights)
            intervention_cost_values.append(-self.temperature*int_cost_tmp)
            intervention_cost_values_original.append(int_cost_tmp)
        
        intervention_cost_values = np.array(intervention_cost_values)

        intervention_cost_values -= intervention_cost_values.max()
        cost_values = [np.exp(c) for c in intervention_cost_values]
        total_cost = np.sum(cost_values)

        probabilities = [(c / total_cost) for c in cost_values]

        # If we need to rescale, then we rescale the probabilities
        probabilities = np.array(probabilities).astype(float)

        return probabilities, intervention_cost_values_original

    def compute_best_action(self, env: EnvironmentWeights, choice_set: list, custom_weights: dict=None) -> list:
        """
        Return the best action for this user, given a choice set
        :param choice_set: choice set with the candidate interventions and action/value
        :param estimated: if True, compute the best action using the estimated weights
        :return: the corresponding best action
        """

        if len(choice_set) == 1:
            return choice_set[0], [0]

        probabilities, intervention_cost_values = self.compute_probabilities(env, choice_set, custom_weights)

        if len(probabilities) > 1:
            choosen_best_action = np.random.choice(len(choice_set), 1, p=probabilities)[0]
        else:
            choosen_best_action = 0

        return choice_set[choosen_best_action], intervention_cost_values

    def compute_choice_probability(self, action, current_env: EnvironmentWeights,
                                   choice_set: list, custom_weights: dict=None,
                                   custom_best_action: list=None) -> float:
        """
        Compute the probability of choosing a given action in a choice set
        :param action: action/value pair we want to get the probability from
        :param choice_set: the choice set from which we need to estimate the probability
        :param estimated: if True, use the estimated weights and not the ground truth
        :return: probability of choosing the given action
        """

        idx = -1
        for k, (a, value, intervn, _, _) in enumerate(choice_set):
            if action[0] == a and action[1] == value and action[2] == intervn:
                idx=k
                break
        assert idx != -1

        probabilities, _ = self.compute_probabilities(current_env, choice_set, custom_weights)

        return probabilities[idx]