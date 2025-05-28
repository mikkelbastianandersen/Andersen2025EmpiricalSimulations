import numpy as np
import random


class SimpleMarket:
    def __init__(self, S_0: np.array, G_0: np.array, mu: np.array, sigma: np.array, alpha: np.array, beta: np.array, rho: np.array, r:float, T: float, dt: float):
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.r = r
        self.T = T
        self.dt = dt

        self.S_0 = S_0
        self.G_0 = G_0

        self.n = len(mu)
        self.steps = int(T / dt)

        self.t = 0

        self.S, self.G = self.simulate()

    def correlated_brownian_motions(self):
        """
        Generate two n-dimensional (take n from shape of mu) correlated Brownian motions, dw and dz. 
        """
        n = len(self.mu)
        # Construct joint covariance matrix
        cov = np.block(
            [
                [
                    np.eye(n),self.rho,
                ],
                [
                    self.rho.T,np.eye(n),
                ],
            ]
        )


        cov = (cov + cov.T) / 2  # Ensure symmetry

        # Add small jitter for numerical stability
        epsilon = 1e-10
        cov += np.eye(cov.shape[0]) * epsilon
        L = np.linalg.cholesky(cov * self.dt)  # Cholesky of joint cov scaled by dt
        eps = np.random.normal(size=2 * n)
        correlated = L @ eps

        dz = correlated[:n]
        dw = correlated[n:]
        return dz, dw
    

    def step(self, S_t: np.array, G_t: np.array):
        self.t += self.dt
        dz, dw = self.correlated_brownian_motions()

        
        returns = self.mu * self.dt + self.sigma @ dz
        esg_impact = self.alpha * self.dt + self.beta @ dw

        S_new = S_t * (returns + 1)
        G_new = G_t + esg_impact

        return S_new, G_new

    def simulate(self):
        S = np.zeros((self.n, self.steps+1))
        G = np.zeros((self.n, self.steps+1))

        S[:, 0] = self.S_0
        G[:, 0] = self.G_0
        for step in range(self.steps):
            S[:, step + 1], G[:, step + 1] = self.step(S[:, step], G[:, step])
            print(f"Step {step + 1}: Current Prices = {S[:, step + 1]}, Current ESG = {G[:, step + 1]}")
        
        return S, G
