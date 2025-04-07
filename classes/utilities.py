import numpy as np

class ExponentialUtility:
    def optimal_portfolio(self, market_state, agent):
        a_j = agent.risk_aversion
        d_j = agent.esg_preference
        wealth = agent.wealth

        mu = market_state['mu']
        alpha = market_state['alpha']
        r = market_state['risk_free_rate']

        Sigma_W = market_state['cov_returns']
        Sigma_Z = market_state['cov_esg']
        Sigma_WZ = market_state['cov_cross']

        M = (a_j ** 2) * Sigma_W + (d_j ** 2) * Sigma_Z + 2 * a_j * d_j * Sigma_WZ
        excess_return = mu - r
        target_vector = a_j * excess_return + d_j * alpha

        try:
            optimal_allocation = np.linalg.solve(M, target_vector) / wealth
        except np.linalg.LinAlgError:
            optimal_allocation = np.zeros_like(mu)

        return optimal_allocation
    

class CRRAUtility:
    def optimal_portfolio(self, market_state, agent):
        # Example logic for CRRA utility (depends on your formula!)
        gamma = agent.risk_aversion
        wealth = agent.wealth

        mu = market_state['mu']
        r = market_state['risk_free_rate']
        Sigma_W = market_state['cov_returns']

        excess_return = mu - r

        try:
            optimal_allocation = (1 / gamma) * np.linalg.solve(Sigma_W, excess_return)
        except np.linalg.LinAlgError:
            optimal_allocation = np.zeros_like(mu)

        return optimal_allocation


