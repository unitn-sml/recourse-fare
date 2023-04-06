from scipy.stats import multivariate_normal, norm
import tqdm

import numpy as np

class MixtureModel:

    def __init__(self, dimensions=10, k=6):

        possible_mus = np.random.randint(-50, 50, size=(k, dimensions))

        self.mixtures_params = []
        for mu_a in possible_mus:
            dim=len(mu_a)
            cov_a = np.zeros((dim, dim), np.float64)
            np.fill_diagonal(cov_a, np.ones(dim)*20)
            self.mixtures_params.append((mu_a, cov_a))
    
    def sample(self, N, mean=False, only_components=False, noise_variance=1, verbose=False):
        samples = []
        for _ in tqdm.tqdm(range(N), disable=not verbose):
            current_sample = []
            for mu, cov in self.mixtures_params:
                current_sample.append(
                    multivariate_normal.rvs(mu, cov, 1) + norm.rvs(0, noise_variance, 1)
                )
            
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

    def logpdf(self, w):
        prob = self.pdf(w)
        return np.log(prob) if prob > 0 else -np.infty

    def pdf(self, w, full=False):

        if full:
            return np.array(
                [multivariate_normal.pdf(w, mu, cov) for mu, cov in self.mixtures_params]
            )

        scaling = len(self.mixtures_params)
        prob = 0.0
        for mu, cov in self.mixtures_params:
            prob += 1/scaling * multivariate_normal.pdf(w, mu, cov)
        return prob