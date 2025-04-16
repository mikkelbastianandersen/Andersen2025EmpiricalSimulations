import numpy as np

class Simulation:
    def __init__(self, market, agents, num_steps):
        self.market = market
        self.agents = agents
        self.num_steps = num_steps
        self.wealth = np.zeros((len(agents), num_steps))
        self.esg = np.zeros((len(agents), num_steps))

    def run(self):
        for step in range(self.num_steps):
            market_before_step = self.market
            market_after_step = self.market.step()
            returns = (market_after_step.prices[:, step + 1] - market_after_step.prices[:, step]) / market_after_step.prices[:, step]
            esg_impacts = market_after_step.esg[:, step + 1] - market_after_step.esg[:, step]
            for i,agent in enumerate(self.agents):
                agent.decide_portfolio(market_before_step)
                
                agent_return = agent.weights @ returns
                agent.wealth *= (1 + agent_return) 
                agent.esg_impact += agent.weights @ esg_impacts

            self.wealth[:, step] = [agent.wealth for agent in self.agents]
            self.esg[:, step] = [agent.esg_impact for agent in self.agents]

        return self.wealth, self.esg