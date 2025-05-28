### An abstract class for utility functions.
from abc import ABC, abstractmethod
import numpy as np

class Utility(ABC):
    def __init__(self):
        pass  # Can be omitted, but here for clarity

    @abstractmethod
    def current_utility(self, X_t, Y_t):
        pass

    @abstractmethod
    def optimal_portfolio(self, X_t, Y_t, mu, sigma, alpha, beta, rho, r):
        pass

    


class CobbDouglasUtility(Utility):
    def __init__(self, a):
        self.a = a

    def current_utility(self, X_t, Y_t):
        return (X_t**self.a) * (Y_t**(1 - self.a))

    def optimal_portfolio(self, X_t, Y_t, mu, sigma, alpha, beta, rho, r):
        """
        Calculate the Cobb-Douglas optimal portfolio allocation.
        Parameters:
            X_t: Current wealth in the risky asset
            Y_t: Current wealth in the risk-free asset
            mu: Expected returns of the risky asset
            sigma: Covariance matrix of the risky asset
            alpha: Expected returns of the risk-free asset
            beta: Covariance matrix of the risk-free asset
            rho: Correlation matrix between the risky and risk-free assets
            r: Risk-free rate
        Returns:
            pi_star: Optimal portfolio allocation
        """
        q_t = X_t / Y_t
        M = (
            self.a * (1 - self.a) * sigma @ sigma.T
            + self.a * (1 - self.a) * q_t**2 * beta @ beta.T
            + 2 * self.a * (self.a - 1) * q_t * sigma @ rho @ beta.T
        )
        M_inv = np.linalg.inv(M)
        vector = (
            self.a * (mu - r)
            + (1 - self.a) * q_t * alpha
        )
        pi_star = M_inv @ vector

        return pi_star
    

class ExponentialUtility(Utility):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def current_utility(self, X_t, Y_t):
        return -self.a * X_t - self.b * Y_t

    def optimal_portfolio(self, X_t, Y_t, mu, sigma, alpha, beta, rho, r):
        """
        Calculate the optimal portfolio allocation.
        Parameters:
            X_t: Current wealth in the risky asset
            Y_t: Current wealth in the risk-free asset
            mu: Expected returns of the risky asset
            sigma: Covariance matrix of the risky asset
            alpha: Expected returns of the risk-free asset
            beta: Covariance matrix of the risk-free asset
            rho: Correlation matrix between the risky and risk-free assets
            r: Risk-free rate
        Returns:
            pi_star: Optimal portfolio allocation
        """
        M = (
            self.a**2 * sigma @ sigma.T
            + self.b**2 * beta @ beta.T
            + 2 * self.a * self.b * sigma @ rho @ beta.T
        )
        M_inv = np.linalg.inv(M)
        vector = self.a * (mu - r) + self.b * alpha
        pi_star = (1/X_t) * M_inv @ vector

        return pi_star