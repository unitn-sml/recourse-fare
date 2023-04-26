import zeus
import numpy as np
import copy

from tqdm import tqdm

from .SliceSampler import SliceSampler
from ...utils.functions import plot_sampler

class SliceSamplerNoiseless(SliceSampler):

    def __init__(self, nodes, mixture, nparticles=100, nsteps=100, temperature=1.5, keep_particles=True, verbose=False):
        
        super().__init__(
            nodes, nparticles, nsteps, temperature, verbose
        )

        self.particles_likelihood = []
        self.keep_particles = keep_particles
        self.mixture = mixture
    
    def log_boundaries(self, w_weights, constraints, env, user):
        if not all([v != 0.0 for v in w_weights]):
            return -np.inf
        w = {k: v for k, v in zip(self.nodes, w_weights)}
        return self.log_likelihood(w, constraints, env, user)

    def log_prior_mu(self, w):
        probab = self.mixture.logpdf(w)
        return probab if np.isfinite(probab) else -np.inf

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

            # compute the best action for this user
            best_action, best_argument, best_intervention, _, _ = choices[md.index(min(md))]

            # Check if the best action has the minimum value
            idx_best_solution = md.index(min(md))
            for idx in range(len(md)):
                if idx == idx_best_solution:
                    continue
                if md[idx] < md[idx_best_solution]:
                    probability *= 0
                    break

            # Compute the probability
            probability *= 1.0 if action[0] == best_action and action[1] == best_argument and action[2] == best_intervention else 0.0

            # If probability is zero break, this weight is not okay
            if probability <= 0:
                break

        assert probability <= 1.0

        return np.log(probability) if probability > 0 else -np.inf

    def logpost(self, w, constraints, env, user):
        lp = self.log_boundaries(w, constraints, env, user)
        if not np.isfinite(lp):
            return -np.inf
        return self.log_prior_mu(w)

    def get_current_particles(self):
        return [{k:v for k,v in zip(self.nodes, w)} for w in self.current_particles]

    def get_mean_majority_particles(self):
        
        particle_class = []
        count_class = {}
        for w in self.current_particles:
            class_probability = self.mixture.pdf(w, full=True)
            idx_best_class = class_probability.argmax()
            
            if idx_best_class in count_class:
                count_class[idx_best_class] += 1
            else:
                count_class[idx_best_class] = 1

            particle_class.append(
                (idx_best_class, w)
            )
        
        majority_class = max(count_class, key=count_class.get)
        particle_class = list(filter(lambda x: x[0]==majority_class, particle_class))
        particle_class = np.array([x[1] for x in particle_class])
        particle_mean = np.mean(particle_class, axis=0).tolist()
        
        return {k:v for k,v in zip(self.nodes, particle_mean)}

    def get_mean_current_particles(self):
        particle_mean = np.mean(self.current_particles, axis=0).tolist()
        return {k:v for k,v in zip(self.nodes, particle_mean)}
    
    def get_mean_high_likelihood_particles(self):

        if len(self.current_particles) == 0:
            print("Empty current particles when getting the mean.")
            return None

        if len(self.particles_likelihood) == 0:
            print("Empty precomputed likelihood when getting the mean.")
            return None

        particles_likelihood = list(zip(
            self.current_particles, self.particles_likelihood
        ))
        particles_likelihood = sorted(particles_likelihood, key=lambda x: x[1], reverse=True)
        particles_likelihood = particles_likelihood[0:25]
        particles_likelihood = [x[0] for x in particles_likelihood]

        particle_mean = np.mean(particles_likelihood, axis=0).tolist()
        return {k:v for k,v in zip(self.nodes, particle_mean)} 

    def sample(self, constraint, env, user, keep=False):

        if constraint is not None and constraint != []:
            self.constraints += constraint
        elif not keep:
            self.constraints = []

        # Generating starting points
        self.start = []

        if self.keep_particles:
            if len(self.current_particles) > 0:
                self.current_particles = list(filter(
                        lambda wx: np.isfinite(self.logpost(wx, self.constraints, copy.deepcopy(env), copy.deepcopy(user))),
                        tqdm(self.current_particles) if self.verbose else self.current_particles
                    ))
                if self.verbose:
                    print(f"\nKeeping {len(self.current_particles)} particles...")
        else:
            self.current_particles = []

        if len(self.current_particles) < self.nparticles:
            max_retries = 100
            while(len(self.current_particles) < self.nparticles and max_retries > 0):
                w_current = self.mixture.sample(self.nparticles)
                w_current = list(filter(
                    lambda wx: np.isfinite(self.logpost(wx, self.constraints, copy.deepcopy(env), copy.deepcopy(user))),
                    tqdm(w_current) if self.verbose else w_current
                ))
                self.current_particles += w_current
                max_retries -= 1
        else:
            w_current = self.current_particles
        
        if self.verbose:
            print("\nFound particles: ", len(self.current_particles))
        
        if len(self.current_particles) < self.nparticles:
            return None, None
        else:
            np.random.shuffle(self.current_particles)
            self.current_particles = self.current_particles[0:self.nparticles]
        
        self.start = self.current_particles

        # 1st step: we sample using the differential move
        sampler = zeus.EnsembleSampler(self.nparticles, self.ndim, self.logpost,
                                           args=[self.constraints, copy.deepcopy(env), copy.deepcopy(user)], light_mode=True,
                                           verbose=False)  # Initialise the sampler
        sampler.run_mcmc(self.start, self.nsteps//2, progress=self.verbose)  # Run sampling

        # Get the burnin samples
        burnin = sampler.get_chain()

        # Set the new starting positions of walkers based on their last positions
        start = burnin[-1]

        # Initialise the Ensemble Sampler using the advanced ``GlobalMove``.
        sampler = zeus.EnsembleSampler(self.nparticles, self.ndim, self.logpost,
                                           args=[self.constraints, copy.deepcopy(env), copy.deepcopy(user)], light_mode=True,
                                           verbose=False, moves=zeus.moves.GlobalMove())
        # Run MCMC
        sampler.run_mcmc(start, self.nsteps//2, progress=self.verbose)

        # Get the particles from the global move
        self.current_particles = sampler.get_chain(discard=self.nsteps // 4, thin=10, flat=False)[0]

        if self.verbose:
            print("CURRENT PARTICLES: ", len(self.current_particles))
            plot_sampler(sampler.get_chain(), self.ndim, sampler=sampler)

        # Compute the likelihood for each particle (for this case is equal to the pdf, in case of the logistic noise is different)
        self.particles_likelihood = [self.logpost(p, self.constraints, copy.deepcopy(env), copy.deepcopy(user)) for p in self.current_particles]

        return sampler, sampler.get_chain(discard=self.nsteps // 2, thin=10)