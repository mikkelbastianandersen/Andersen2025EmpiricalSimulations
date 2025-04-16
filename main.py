import numpy as np
import matplotlib.pyplot as plt

from classes.assets import Asset
from classes.agents import Agent
from classes.portfolios import Portfolio
from classes.utilities import ExponentialUtility
from classes.market import Market
from classes.simulation import Simulation

# --- Reproducibility ---
np.random.seed(42)

# --- Define Assets ---
asset1 = Asset(name='Asset1', mu_0=0.05, sigma=np.array([0.2, 0.4]), alpha=0.01, beta=np.array([0.05, 0.02]))
asset2 = Asset(name='Asset2', mu_0=0.07, sigma=np.array([0.3, 0.1]), alpha=-0.005, beta=np.array([0.03, 0.01]))
assets = [asset1, asset2]

# --- Define Market ---
corr_cross = np.array([
    [0.3, -0.2],
    [0.1, 0.4]
])
market = Market(assets=assets, corr_cross=corr_cross, lambda_esg=0.5, dt=1/252)

# --- Define Agents ---
agent1 = Agent(name='Agent1', initial_wealth=1000, risk_aversion=3, esg_preference=1, utility=ExponentialUtility())
agent2 = Agent(name='Agent2', initial_wealth=1000, risk_aversion=2, esg_preference=2, utility=ExponentialUtility())

# --- Assign Portfolios ---
agent1.portfolio = Portfolio(assets)
agent2.portfolio = Portfolio(assets)
agents = [agent1, agent2]

# --- Run Simulation ---
simulation = Simulation(market=market, agents=agents, num_steps=10000)
wealth, esg = simulation.run()

# --- Plot Results ---
time = np.arange(10000)

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
