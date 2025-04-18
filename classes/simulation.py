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
            self.market.step()
            self.market.update_beliefs()

            self.wealth[:, step] = [agent.wealth for agent in self.agents]
            self.esg[:, step] = [agent.esg_impact for agent in self.agents]

        return self.wealth, self.esg