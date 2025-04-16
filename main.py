import numpy as np
import matplotlib.pyplot as plt

from classes.agents import Agent
from classes.portfolios import Portfolio
from classes.utilities import ExponentialUtility, LogLogUtility
from classes.market import Market
from classes.simulation import Simulation

# --- Reproducibility ---
np.random.seed(42)

num_steps = 1000

# --- Define Agents ---
agent1 = Agent(name='Agent1', initial_wealth=1, utility=LogLogUtility(esg_preference=1))
agent2 = Agent(name='Agent2', initial_wealth=1, utility=ExponentialUtility(risk_aversion=0.1, esg_preference=3))
agents = [agent1, agent2]

# --- Define Market ---
mu_0 = np.array([0.05, 0.07])
sigma_0 = np.array([[0.2, 0.3], [0.3, 0.4]])
alpha_0 = np.array([0.01, 0.008])
beta_0 = np.array([[0.1, 0.2], [0.02, 0.01]])

rho_0 = np.array([
    [0.3, -0.2],
    [0.1, 0.4]
])
market = Market(mu_t=mu_0, sigma_t=sigma_0, alpha_t=alpha_0, beta_t=beta_0, rho_t=rho_0, r= 0, agents = agents, num_steps=num_steps, dt=1/1000)


# --- Run Simulation ---
simulation = Simulation(market=market, agents=agents, num_steps=num_steps)
wealth, esg = simulation.run()

# --- Plot Results ---
time = np.arange(num_steps)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(time, wealth[0], label='Agent 1')
plt.plot(time, wealth[1], label='Agent 2')
plt.title('Wealth Over Time')
plt.xlabel('Time Step')
plt.ylabel('Wealth')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time, esg[0], label='Agent 1 ESG')
plt.plot(time, esg[1], label='Agent 2 ESG')
plt.title('Cumulative ESG Impact')
plt.xlabel('Time Step')
plt.ylabel('ESG Impact')
plt.legend()

plt.tight_layout()
plt.show()
