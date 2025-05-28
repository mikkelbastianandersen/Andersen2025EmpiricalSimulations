""" A class to represent an agent that walks through a simple simulated market (constant parameters). """
from simple_market import SimpleMarket
from utility import Utility
import numpy as np


class AgentInSimpleMarket:
    def __init__(self, utility: Utility, market: SimpleMarket, initial_wealth: float = 100, initial_esg_impact: float = 100):
        """
        Initialize the agent with a utility function and a market simulator.
        
        Parameters:
            utility: An instance of a utility class (e.g., CobbDouglasUtility).
            simulator: An instance of SimpleMarketSimulator.
        """
        self.market = market # market parameters
        self.current_step = 0
        self.utility = utility

        self.X_0 = initial_wealth  # Initial wealth for the agent
        self.Y_0 = initial_esg_impact # Initial ESG impact for the agent
        

        self.X, self.Y, self.utility_ts = self.walk()
    
    def walk(self):
        """
        Simulate the agent's walk through the market.
        The agent will make decisions based on the utility function and the current market state.
        """
        X = np.zeros(self.market.steps+1)
        Y = np.zeros(self.market.steps+1)
        utility_ts = np.zeros(self.market.steps+1)
        X[0] = self.X_0
        Y[0] = self.Y_0
        utility_ts[0] = self.utility.current_utility(X[0], Y[0])

        for step in range(self.market.steps):
            # Get current asset prices and ESG impacts
            X_t = X[step]
            Y_t = Y[step]


            # Calculate the utility at the current state
            current_utility = self.utility.current_utility(X_t, Y_t)
            print(f"Step {step}: Current Utility = {current_utility}")

            # Calculate the optimal portfolio allocation
            pi_star = self.utility.optimal_portfolio(X_t, Y_t, self.market.mu, self.market.sigma, self.market.alpha, self.market.beta, self.market.rho, self.market.r)
            pi_r = 1-sum(pi_star)
            print(f"Step {step}: Optimal Portfolio Allocation to risky assets = {pi_star}" )
            print(f"Step {step}: Optimal Portfolio Allocation to risk-free asset = {pi_r}" )

            # Calculate returns and ESG impact for the next step
            returns = self.market.S[:, step + 1] / self.market.S[:, step] - 1
            G_t = self.market.G[:, step+1] - self.market.G[:, step]

            # Update wealth and ESG impact based on the optimal portfolio allocation
            X_new = X_t * (pi_star @ (1+returns)) + X_t * pi_r * (1 + self.market.r * self.market.dt)  # Wealth update
            Y_new = Y_t + G_t @ pi_star
            X[step + 1] = X_new
            Y[step + 1] = Y_new
            utility_ts[step + 1] = self.utility.current_utility(X_new, Y_new)
            print(f"Step {step + 1}: New Wealth = {X_new}, New ESG Impact = {Y_new}")
        return X, Y, utility_ts
            

    
        