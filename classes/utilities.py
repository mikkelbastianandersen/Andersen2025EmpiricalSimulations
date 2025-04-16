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
        rho = market_state.rho_t
        r = market_state.r

        Sigma_W = sigma @ sigma.T
        Sigma_Z = beta @ beta.T
        Sigma_WZ = sigma @ rho @ beta.T

        M = (a ** 2) * Sigma_W + (d ** 2) * Sigma_Z + 2 * a * d * Sigma_WZ
        M_inv = np.linalg.inv(M)
        target = a * (mu - r) + d * alpha
        weights = 1/wealth * M_inv @ target

        return weights



class LogLogUtility:
    def __init__(self, esg_preference):
        """
        Log utility over wealth and ESG impact:
        u(X, Y) = log(X) + B * log(Y)

        Parameters:
        - esg_preference (B): weight on log(Y)
        """
        self.B = esg_preference

    def optimal_portfolio(self, agent, market_state):
        """
        Compute the closed-form optimal portfolio weights under log-log utility.

        Parameters:
        - agent: must have .wealth and .y (ESG impact level)
        - market_state: must have .mu_t, .alpha_t, .sigma_t, .beta_t, .rho_t, .r

        Returns:
        - weights: optimal portfolio weights (numpy array)
        """
        B = self.B
        y = agent.esg_impact
        y_safe = max(y, 0.05) # Impose baseline for ESG impact

        mu = market_state.mu_t
        alpha = market_state.alpha_t
        sigma = market_state.sigma_t
        beta = market_state.beta_t
        rho = market_state.rho_t
        r = market_state.r

        Sigma_W = sigma @ sigma.T
        Sigma_Z = beta @ beta.T
        Sigma_WZ = sigma @ rho @ beta.T

        # Full adjusted risk matrix
        M = Sigma_W + (B / y_safe**2) * Sigma_Z + (2 * B / y_safe) * Sigma_WZ
        M_inv = np.linalg.inv(M)

        # Tilt vector (financial + ESG preference)
        target = (mu - r) + (B / y_safe) * alpha

        weights = M_inv @ target
        return weights

    

