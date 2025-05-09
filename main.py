from core.agents.agent import Agent
from core.agents.utilities.exponential_utility import ExponentialUtility
from simulation import Simulation
import pandas as pd
import config

sim = Simulation(config)
collector = sim.run()

agent = Agent(
    utility=ExponentialUtility(a=0.5, d=3.0),
    initial_wealth=100.0
)

for t in range(collector.T):
    mu_t = collector.mu[t]
    sigma_t = collector.sigma[t]
    alpha_t = collector.alpha[t]
    beta_t = collector.beta[t]
    rho_t = collector.rho[t]
    returns_t = collector.returns[t]
    esg_t = collector.esg[t]


    agent.decide_portfolio(mu_t, sigma_t, alpha_t, beta_t, rho_t, config.r)
    # print(f"Portfolio at time {t}: {agent.portfolio}")
    
    
    agent.update(returns_t, esg_t)
    # print(f"Wealth at time {t}: {agent.wealth}")
    # print(f"ESG score at time {t}: {agent.esg}")


print(f"Final wealth: {agent.wealth}")
print(f"Final ESG score: {agent.esg}")
