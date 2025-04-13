import numpy as np
from classes.market import Market
from classes.agents import Agent
from classes.portfolios import Portfolio
from classes.assets import Asset

class Simulation:
    def __init__(self, market: Market, agents: list[Agent], num_steps: int):
        """
        Initialize the Simulation class with parameters.
        """
        self.N = len(market.assets)
        self.M = len(agents)
        self.num_steps = num_steps
        self.dt = market.dt
        self.time = 0

        self.agents = agents
        self.market = market

        self.wealth_over_time = np.zeros((self.M, self.num_steps))
        self.esg_impact_over_time = np.zeros((self.M, self.num_steps))


    def run(self):
        for step in range(self.num_steps):
            self.time += self.dt
            self.wealth_over_time[:, step] = np.array([agent.portfolio.get_wealth() for agent in self.agents])
            self.esg_impact_over_time[:, step] = np.array([agent.portfolio.get_esg_impact() for agent in self.agents])

        return {
            'wealth_over_time': self.wealth_over_time,
            'esg_impact_over_time': self.esg_impact_over_time
        }

    