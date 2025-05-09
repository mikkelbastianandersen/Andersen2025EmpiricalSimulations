import numpy as np


class CorrelatedNoise:
    def __init__(self, sigma_t, beta_t, rho_t):
        self.sigma_t = sigma_t  # shape: (n, n)
        self.beta_t = beta_t  # shape: (n, n)
        self.rho_t = rho_t  # shape: (n, n)

    def sample(self, dt):
        n = self.sigma_t.shape[0]
        # Construct joint covariance matrix
        cov = np.block(
            [
                [
                    self.sigma_t @ self.sigma_t.T,
                    self.sigma_t @ self.rho_t @ self.beta_t.T,
                ],
                [
                    self.beta_t @ self.rho_t.T @ self.sigma_t.T,
                    self.beta_t @ self.beta_t.T,
                ],
            ]
        )
        cov = (cov + cov.T) / 2  # Ensure symmetry

        # Add small jitter for numerical stability
        epsilon = 1e-10
        cov += np.eye(cov.shape[0]) * epsilon
        L = np.linalg.cholesky(cov * dt)  # Cholesky of joint cov scaled by dt
        eps = np.random.normal(size=2 * n)
        correlated = L @ eps

        dz = correlated[:n]
        dw = correlated[n:]
        return dz, dw
