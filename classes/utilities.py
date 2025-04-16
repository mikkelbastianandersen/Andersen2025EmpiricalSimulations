import numpy as np

class ExponentialUtility:
    def optimal_portfolio(self, agent, market_state):
        a = agent.risk_aversion
        d = agent.esg_preference
        wealth = agent.wealth

        mu = market_state['mu']
        alpha = market_state['alpha']
        sigma = market_state['sigma']
        beta = market_state['beta']

        Sigma_W = sigma @ sigma.T
        Sigma_Z = beta @ beta.T
        Sigma_WZ = sigma @ beta.T

        M = (a ** 2) * Sigma_W + (d ** 2) * Sigma_Z + 2 * a * d * Sigma_WZ
        target = a * (mu - market_state['risk_free_rate']) + d * alpha

        try:
            weights = np.linalg.solve(M, target) / wealth
        except np.linalg.LinAlgError:
            weights = np.zeros_like(mu)

        return weights