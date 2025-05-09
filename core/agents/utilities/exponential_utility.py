import numpy as np
from core.agents.utilities.utility import UtilityFunction

class ExponentialUtility(UtilityFunction):
    def __init__(self, a, d):
        self.a = a
        self.d = d

    def evaluate(self, wealth, esg_impact):
        return -np.exp(-self.a * wealth - self.d * esg_impact)

    def optimal_portfolio(self, mu, sigma, alpha, beta, rho, r, wealth):
        # closed-form solution: w* = (1/X)*M^(-1)*(a*(mu-r*1)+d*alpha)
        # where M = a^2 sigma*simga^T + d^2 beta*beta^T + 2*a*d*sigma*rho*beta^T

        M = self.a ** 2 * sigma @ sigma.T + self.d ** 2 * beta @ beta.T + 2 * self.a * self.d * sigma @ rho @ beta.T
        M_inv = np.linalg.inv(M)

        w = 1/wealth * M_inv @ (self.a * (mu - r * np.ones(mu.shape)) + self.d * alpha)
        return w
