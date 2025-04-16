import numpy as np

class Market:
    def __init__(self, mu_t, sigma_t, alpha_t, beta_t, rho_t, dt=1/252):
        self.dt = dt
        self.mu_t = mu_t
        self.sigma_t = sigma_t
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.rho_t = rho_t

        self.N = len(mu_t)
        self.K = len(sigma_t)

        self.esg = np.zeros(self.N)
        self.prices = np.ones(self.N)

    def build_joint_loading(self):
        I = np.eye(self.alpha_tN)
        joint_corr = np.block([
            [I,        self.rho_t],
            [self.rho_t.T,  I     ]
        ])  # shape (2N, 2N)
        L = np.linalg.cholesky(joint_corr)
        return L

    def step(self):
        # Generate correlated shocks for returns and ESG
        L = self.build_joint_loading()
        stand_norms = np.random.randn(2 * self.N)
        joint_shocks = L @ stand_norms
        z = joint_shocks[:self.N]
        w = joint_shocks[self.N:]

        # Save previous prices and ESG for return calculation
        esg_before = self.esg.copy()
        prices_before = self.prices.copy()

        # Update ESG for each asset
        self.esg += self.alpha * self.dt + np.sqrt(self.dt) * self.beta_t @ w

        # Update prices for each asset
        returns = self.mu_t * self.dt + np.sqrt(self.dt) * self.sigma_t @ z
        self.prices *= returns
        self.prices = np.maximum(self.prices, 1e-6)

        # Update \mu_t
        mu = self.mu_t + np.random.randn(self.N)
        

        return {
            'returns': returns,
            'esg': self.esg,
            'mu': mu,
            'alpha': self.alpha,
            'sigma': self.sigma,
            'beta': self.beta,
            'risk_free_rate': 0.0
        }