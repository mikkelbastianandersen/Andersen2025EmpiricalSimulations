# main.py

from classes.assets import Asset
from classes.market import Market
from classes.portfolios import Portfolio
from classes.agents import Agent
from classes.utilities import ExponentialUtility, CRRAUtility
from classes.simulation import Simulation

import numpy as np
import matplotlib.pyplot as plt

# --- Set random seed for reproducibility ---
np.random.seed(42)

# --- Define assets ---
asset1 = Asset(name='Asset1', mu=0.05, sigma=np.array([0.2,0.4]), alpha=0.01, beta=np.array([0.2,0.4]))
asset2 = Asset(name='Asset2', mu=0.07, sigma=np.array([0.3,0.3]), alpha=-0.005, beta=np.array([0.3,0.3]))

assets = [asset1, asset2]

# --- Correlation matrix for the two standard Brownian motions ---
corr_cross = np.array([[0.3, -0.2],
                       [0.1, 0.4]])

# --- Initialize market ---
market = Market(
    assets=assets,
    corr_cross=corr_cross,
    dt=1/252
)


# --- Initialize agent(s) ---
agent1 = Agent(
    initial_wealth=1000,
    risk_aversion=3,
    esg_preference=1,
    utility=ExponentialUtility()
)

agent2 = Agent(
    initial_wealth=1000,
    risk_aversion=3,
    esg_preference=1,
    utility=ExponentialUtility()
)

agents = [agent1, agent2]

# --- Simulation settings ---
num_steps = 252  # one year


# --- Run market simulation ---
sim = Simulation(market=market, agents=agents, num_steps=num_steps)
sim_result = sim.run()
# --- Extract results ---
wealth_over_time = sim_result['wealth_over_time']
esg_impact_over_time = sim_result['esg_impact_over_time']
# --- Plot settings ---
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# --- Plot results ---
time = np.arange(num_steps)

plt.figure(figsize=(12, 5))

# Wealth plot
plt.subplot(1, 2, 1)
plt.plot(time, wealth_over_time, label='Agent Wealth')
plt.title('Wealth Over Time')
plt.xlabel('Time Step')
plt.ylabel('Wealth')
plt.legend()

# ESG impact plot
plt.subplot(1, 2, 2)
plt.plot(time, esg_impact_over_time, label='Agent ESG Impact', color='green')
plt.title('Cumulative ESG Impact')
plt.xlabel('Time Step')
plt.ylabel('ESG Impact')
plt.legend()

plt.tight_layout()
plt.show()
