from scipy.stats import multivariate_normal, norm
import tqdm

import numpy as np

class MixtureModel:

    def __init__(self, dimensions=10, k=6, mixture_means: list = None):

        if mixture_means:
            possible_mus = mixture_means
        else:
            possible_mus = np.random.randint(-50, 50, size=(k, dimensions))

        self.mixtures_params = []
        for mu_a in possible_mus:
            dim=len(mu_a)
            cov_a = np.zeros((dim, dim), np.float64)
            np.fill_diagonal(cov_a, np.ones(dim))
            self.mixtures_params.append((mu_a, cov_a))
    
    def sample(self, N, mean=False, only_components=False, noise_variance=1, verbose=False):
        samples = []

        current_samples = []
        for mu, cov in self.mixtures_params:
            current_samples.append(
                multivariate_normal.rvs(mu, cov, N)
            )
        
        for current_sample in tqdm.tqdm(zip(*current_samples), disable=not verbose):
            
            if mean:
                samples.append(
                    np.mean(current_sample, axis=0)
                )
            elif only_components:
                 samples.append(
                    current_sample
                )
            else:
                samples.append(
                    current_sample[np.random.choice(len(current_sample))]
                )
        return samples

    def logpdf(self, w, cov_scaling: float=1.0):
        prob = self.pdf(w, cov_scaling=cov_scaling)
        return np.log(prob) if prob > 0 else -np.infty

    def pdf(self, w, full=False, cov_scaling: float=1.0):

        if full:
            return np.array(
                [multivariate_normal.pdf(w, mu, cov*cov_scaling) for mu, cov in self.mixtures_params]
            )

        scaling = len(self.mixtures_params)
        prob = 0.0
        for mu, cov in self.mixtures_params:
            prob += 1/scaling * multivariate_normal.pdf(w, mu, cov*cov_scaling)
        return prob