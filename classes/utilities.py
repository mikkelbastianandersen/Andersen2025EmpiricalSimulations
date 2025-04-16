import numpy as np

class ExponentialUtility:
    def __init__(self, risk_aversion, esg_preference):
        self.risk_aversion = risk_aversion
        self.esg_preference = esg_preference
        
    def optimal_portfolio(self, agent, market_state):
        a = self.risk_aversion
        d = self.esg_preference
        wealth = agent.wealth

        mu = market_state.mu_t
        alpha = market_state.alpha_t
        sigma = market_state.sigma_t
        beta = market_state.beta_t
        r = market_state.r

        Sigma_W = sigma @ sigma.T
        Sigma_Z = beta @ beta.T
        Sigma_WZ = sigma @ beta.T

        M = (a ** 2) * Sigma_W + (d ** 2) * Sigma_Z + 2 * a * d * Sigma_WZ
        M_inv = np.linalg.inv(M)
        target = a * (mu - r) + d * alpha
        weights = -1/wealth * M_inv @ target

        return weights