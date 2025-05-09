import numpy as np


class DataCollector:
    def __init__(self, T, n_assets):
        self.T = T
        self.n = n_assets

        self.returns = np.zeros((T, n_assets))
        self.prices = np.zeros((T + 1, n_assets))
        self.esg = np.zeros((T + 1, n_assets))
        self.mu = np.zeros((T, n_assets))
        self.sigma = np.zeros((T, n_assets, n_assets))
        self.beta = np.zeros((T, n_assets, n_assets))
        self.rho = np.zeros((T, n_assets, n_assets))

        self.t = 0

    def record(self, r_t, S_t, G_t, mu_t, sigma_t, beta_t, rho_t):
        self.returns[self.t] = r_t
        self.prices[self.t + 1] = S_t
        self.esg[self.t + 1] = G_t
        self.mu[self.t] = mu_t
        self.sigma[self.t] = sigma_t
        self.beta[self.t] = beta_t
        self.rho[self.t] = rho_t
        self.t += 1
