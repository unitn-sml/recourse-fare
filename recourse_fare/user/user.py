from abc import abstractmethod, ABC

from ..environment_w import EnvironmentWeights
from ..utils.wfare_utils.structural_weights import StructuralWeights

import numpy as np



class User(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute_best_action(self, current_env: EnvironmentWeights, choice_set: list, custom_weights: dict=None):
        pass

    @abstractmethod
    def compute_choice_probability(self, action, value, current_env: EnvironmentWeights, choice_set: list, custom_weights: dict=None) -> float:
        pass

    def compute_intervention_cost(self, env: EnvironmentWeights, env_state: dict, intervention: list, custom_weights: dict=None) -> float:
        """
        Compute the cost of an intervention. It does not modify the environment, however, this
        function is unsafe to be used in a multi-threaded context. Unless the object in replicated
        in each process separately
        :param intervention: ordered list of action/value/type tuples
        :param custom_env: feature updates which are applied before computing the cost (not persistent)
        :param estimated: if True, we compute the cost using the estimated graph
        :return: intervention cost
        """

        prev_state = env.features.copy()
        prev_weights = env.weights.copy()

        env.features = env_state.copy()
        env.weights = custom_weights if custom_weights else env.weights

        intervention_cost = 0
        for action, value in intervention:
            prog_idx = env.prog_to_idx.get(action)
            value_idx = env.complete_arguments.index(value)
            intervention_cost += env.get_cost(prog_idx, value_idx)
            env.act(action, value)

        env.features = prev_state
        env.weights = prev_weights

        return intervention_cost

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
            return choice_set[0]

        intervention_cost_values = []
        for _, _, intervention, current_env, _ in choice_set:
            int_cost_tmp = self.compute_intervention_cost(env, current_env, intervention, custom_weights)
            intervention_cost_values.append(int_cost_tmp)

        return choice_set[intervention_cost_values.index(min(intervention_cost_values))]

    def compute_choice_probability(self, action, current_env: EnvironmentWeights, choice_set: list, custom_weights: dict=None) -> float:
        """
        Compute the probability of choosing a given action in a choice set
        :param action: action/value pair we want to get the probability from
        :param choice_set: the choice set from which we need to estimate the probability
        :param estimated: if True, use the estimated weights and not the ground truth
        :return: probability of choosing the given action
        """
        best_action, best_argument, best_intervention, _, _ = self.compute_best_action(current_env, choice_set, custom_weights)
        return 1.0 if action[0] == best_action and action[1] == best_argument and action[2] == best_intervention else 0.0

class LogisticNoiseUser(User):

    def __init__(self, temperature: float=0.9):
        self.temperature = temperature
        super().__init__()

    def compute_probabilities(self, env, choice_set, custom_weights):

        intervention_cost_values = []
        for _, _, intervention, current_env, _ in choice_set:
            int_cost_tmp = self.compute_intervention_cost(env, current_env, intervention, custom_weights)
            intervention_cost_values.append(-self.temperature*int_cost_tmp)
        
        intervention_cost_values = np.array(intervention_cost_values)

        intervention_cost_values -= intervention_cost_values.max()
        cost_values = [np.exp(c) for c in intervention_cost_values]
        total_cost = np.sum(cost_values)

        probabilities = [(c / total_cost) for c in cost_values]

        # If we need to rescale, then we rescale the probabilities
        probabilities = np.array(probabilities).astype(float)

        return probabilities

    def compute_best_action(self, env: EnvironmentWeights, choice_set: list, custom_weights: dict=None) -> list:
        """
        Return the best action for this user, given a choice set
        :param choice_set: choice set with the candidate interventions and action/value
        :param estimated: if True, compute the best action using the estimated weights
        :return: the corresponding best action
        """

        if len(choice_set) == 1:
            return choice_set[0]

        probabilities = self.compute_probabilities(env, choice_set, custom_weights)

        if len(probabilities) > 1:
            choosen_best_action = np.random.choice(len(choice_set), 1, p=probabilities)[0]
        else:
            choosen_best_action = 0

        return choice_set[choosen_best_action]

    def compute_choice_probability(self, action, current_env: EnvironmentWeights, choice_set: list, custom_weights: dict=None) -> float:
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

        probabilities = self.compute_probabilities(current_env, choice_set, custom_weights)

        return probabilities[idx]