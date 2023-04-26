import zeus
import numpy as np
import copy

from tqdm import tqdm

from .SliceSamplerNoiseless import SliceSamplerNoiseless
from ...utils.functions import plot_sampler

class SliceSamplerLogistic(SliceSamplerNoiseless):

    def __init__(self, nodes, mixture, nparticles=100, nsteps=100, temperature=0.9, keep_particles=True, verbose=False):
        super().__init__(nodes, mixture, nparticles, nsteps, temperature, keep_particles, verbose)

        self.temperature = 0.9

    def log_likelihood(self, w, constraints, env, user):

        # Initialize probability
        probability = 1

        # Evaluate the consistency of these weights with the linear constraints
        for action, choices in constraints:

            # No other choices, the user picks only one action
            if len(choices) == 1:
                continue
            
            # Compute the cost given the weights
            md = [user.compute_intervention_cost(env, current_env, intervention, w) for a, k, intervention, current_env, _ in choices]

            # Get intervention of the best choice
            best_action_idx = -1
            for i, v in enumerate(choices):
                if v[0] == action[0] and v[1] == action[1] and action[2] == v[2]:
                    best_action_idx = i
                    break
            assert best_action_idx != -1
            
            # Compute the best action for this user
            md = -self.temperature * np.array(md)
            md = md - md.max()
            md = [np.exp(m) for m in md]
            total_cost = np.sum(md)
            probabilitiy_choices = [(c / total_cost) for c in md]

            # If probability is 0, break
            if probabilitiy_choices[best_action_idx] <= 0.0:
                probability = -np.inf
                break

            probability += np.log(probabilitiy_choices[best_action_idx])

        return probability if probability > 0 else -np.inf

    def logpost(self, w, constraints, env, user):
        lp = self.log_boundaries(w, constraints, env, user)
        if not np.isfinite(lp):
            return -np.inf
        return self.log_prior_mu(w) + lp