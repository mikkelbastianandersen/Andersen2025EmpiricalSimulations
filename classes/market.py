# market.py

import numpy as np

class Market:
    def __init__(self, assets, corr_cross, dt=1/252):
        self.assets = assets
        self.asset_names = [asset.name for asset in assets]
        self.N = len(assets)
        self.dt = dt
        self.time = 0

        # Parameters
        self.mu = np.array([asset.mu for asset in assets])
        self.sigma = np.matrix([asset.sigma for asset in assets])

        self.alpha = np.array([asset.alpha for asset in assets])
        self.beta = np.array([asset.beta for asset in assets])

        # Initial values
        self.prices = np.ones(self.N)
        self.esg_impacts = np.zeros(self.N)

        # Build the full covariance matrix
        cov_returns = self._correlation_to_covariance(corr_returns, self.sigma)
        cov_esg = self._correlation_to_covariance(corr_esg, np.ones(self.N))  # BM has unit volatility
        cov_cross = self._build_cross_covariance(corr_cross, self.sigma)

        # Assemble full covariance matrix
        self.full_cov = np.block([
            [cov_returns, cov_cross],
            [cov_cross.T, cov_esg]
        ])
        self.full_cov = ensure_psd(self.full_cov)
        self.L = np.linalg.cholesky(self.full_cov)

    def _correlation_to_covariance(self, corr, stddevs):
        return np.outer(stddevs, stddevs) * corr

    def _build_cross_covariance(self, corr_cross, stddevs_returns):
        return np.outer(stddevs_returns, np.ones(self.N)) * corr_cross


    def evolve_market(self):
        # Generate uncorrelated normals
        Z = np.random.normal(0, 1, size=2 * self.N)

        # Correlated shocks
        correlated_Z = self.L @ Z

        # Split shocks
        Z_returns = correlated_Z[:self.N]
        Z_esg = correlated_Z[self.N:]

        # Update prices (GBM)
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z_returns
        self.prices *= np.exp(drift + diffusion)

        # Prepare returns as percentage changes
        returns = {name: change for name, change in zip(self.asset_names, np.exp(drift + diffusion) - 1)}

        # Update ESG impacts (standard BM)
        self.esg_impacts += (
            self.alpha * self.dt +
            self.beta * np.sqrt(self.dt) * Z_esg
        )
        esg_scores = {name: impact for name, impact in zip(self.asset_names, self.esg_impacts)}

        self.time += 1

        return {
            'returns': returns,
            'esg': esg_scores,
            'mu': self.mu,
            'alpha': self.alpha,
            'risk_free_rate': 0.0,  # or set your desired risk-free rate
            'cov_returns': self.full_cov[:self.N, :self.N],
            'cov_esg': self.full_cov[self.N:, self.N:],
            'cov_cross': self.full_cov[:self.N, self.N:]
        }

    def get_state(self):
        return self.evolve_market()


def ensure_psd(matrix, tol=1e-8):
    """Ensure the matrix is positive semidefinite."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < tol] = tol  # set small/negative eigenvalues to threshold
    return eigvecs @ np.diag(eigvals) @ eigvecs.T