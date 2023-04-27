import zeus
import numpy as np
import copy

from scipy.stats import multivariate_normal, norm, uniform
from copy import deepcopy

from ...utils.functions import compute_intervention_cost

class SliceSampler:

    def __init__(self, nodes, nparticles=100, nsteps=100, temperature=1.5, verbose=True):

        self.nodes = nodes

        self.verbose = verbose

        self.nparticles = nparticles

        self.ndim = len(nodes)  # Number of parameters/dimensions (mean and sigma)
        self.nwalkers = self.ndim*2  # Number of walkers to use. It should be at least twice the number of dimensions.
        self.nsteps = nsteps  # Number of steps/iterations.

        self.constraints = []

        self.current_particles = []

        self.temperature = temperature

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        return result


    def log_prior_mu(self, w):
        """
        Log-prior distribution over the mean. We know in advance the covariance matrix
        since we assume that the weights are uncorrelated with each other.
        :param mu: the estimated mu
        :param mu_zero: the mean of the prior
        :return: log pdf
        """
        return uniform.logpdf(w[0], 1, 100)

    def log_likelihood(self, w, constraints, env):

        # Evaluate the consistency of these weights with the linear constraints
        probability = 1
        for best_action, choices in constraints:

            # No other choices, the user picks only one action
            if len(choices) == 1:
                continue

            # Get intervention of the best choice
            best_action_env = -1
            for i, v in enumerate(choices):
                if v[0] == best_action[0] and v[1] == best_action[1]:
                    best_action_env = i
                    break
            assert best_action_env != -1

            md = np.array([-self.temperature * compute_intervention_cost(env, current_env, intervention, w) for a, k, intervention, current_env, _ in choices])
            md = md - md.max()
            md = [np.exp(m) for m in md]
            total_cost = np.sum(md)

            probabilitiy_choices = [(c / total_cost) for c in md]

            probability *= probabilitiy_choices[best_action_env]

            # If probability is zero break, this weight is not okay
            if probability <= 0:
                break

        assert probability <= 1.0

        return np.log(probability) if probability > 0 else -1000

    def logpost(self, w_weights, constraints, env):
        '''The natural logarithm of the posterior.'''
        w = {k: v for k, v in zip(self.nodes, w_weights)}
        return self.log_prior_mu(w_weights) + self.log_likelihood(w, constraints, env)

    def get_current_particles(self):
        return [{k:v for k,v in zip(self.nodes, w)} for w in self.current_particles]

    def get_mean_current_particles(self):
        particle_mean = np.mean(self.current_particles, axis=0).tolist()
        return {k:v for k,v in zip(self.nodes, particle_mean)}

    def reset_constraints(self):
        self.constraints = []

    def sample(self, constraint, env, user, questions=1):

        assert constraint
        self.constraints += constraint

        # Generating starting points
        if self.constraints:
            self.start = []
            not_found = 5000  # Tolerance, if we are not able to generate valid
                              # walker positions, then we abort for this user.
            counter = self.nwalkers-1
            while counter >= 0 and not_found > 0:
                w = np.random.multivariate_normal(self.mu_zero, self.cov_zer, 1)[0]

                if np.isfinite(self.logpost(w, self.constraints, copy.deepcopy(env))):
                    self.start.append(w)
                    counter -= 1
                    not_found = 5000
                else:
                    not_found -= 1

            if not_found <= 0 and counter > 0:
                raise ValueError("I was not able to initialize walkers for this user.")

        # Based on how many questions we have, we run the sampler for more time
        current_steps_needed = self.nsteps * questions

        sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.logpost,
                                           args=[self.constraints, copy.deepcopy(env)], verbose=False
                                           )  # Initialise the sampler

        sampler.run_mcmc(self.start, current_steps_needed, progress=True)  # Run sampling

        self.current_particles = sampler.get_chain(discard=self.nsteps // 2, thin=10, flat=False)[0]

        return sampler, sampler.get_chain(discard=self.nsteps // 2, thin=10)