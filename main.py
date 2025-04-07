# main.py

from classes.assets import Asset
from classes.market import Market
from classes.portfolios import Portfolio
from classes.agents import Agent
from classes.utilities import ExponentialUtility, CRRAUtility

import numpy as np
import matplotlib.pyplot as plt

# --- Set random seed for reproducibility ---
np.random.seed(42)

# --- Define assets ---
asset1 = Asset(name='Asset1', mu=0.05, sigma=0.2, alpha=0.01, beta=0.05)
asset2 = Asset(name='Asset2', mu=0.07, sigma=0.25, alpha=-0.005, beta=0.03)

assets = [asset1, asset2]

# --- Correlation matrices ---
corr_returns = np.array([[1.0, 0.8],
                         [0.8, 1.0]])

corr_esg = np.array([[1.0, 0.4],
                     [0.4, 1.0]])

corr_cross = np.array([[0.3, -0.2],
                       [0.1, 0.4]])

# --- Initialize market ---
market = Market(
    assets=assets,
    corr_returns=corr_returns,
    corr_esg=corr_esg,
    corr_cross=corr_cross,
    dt=1/252
)

# --- Initialize utility function ---
# You can switch between ExponentialUtility() or CRRAUtility()
utility_function = ExponentialUtility()
# utility_function = CRRAUtility()

# --- Initialize agent ---
agent = Agent(
    initial_wealth=1000,
    risk_aversion=3,
    esg_preference=1,
    utility=utility_function
)

# --- Initialize portfolio ---
initial_allocations = {asset.name: 0.5 for asset in assets}  # equal weights
agent.portfolio = Portfolio(assets=assets)
agent.portfolio.update_holdings(initial_allocations)

# --- Simulation settings ---
num_steps = 252  # one year

# --- Storage for results ---
wealth_over_time = []
esg_impact_over_time = []
prices_over_time = {asset.name: [] for asset in assets}

# --- Run simulation ---
for step in range(num_steps):
    state = market.get_state()

    # Agent optimizes portfolio using injected utility function
    agent.optimize_portfolio(state)

    # Agent updates wealth and ESG impact
    agent.update(state)

    # Record results
    wealth_over_time.append(agent.wealth)
    esg_impact_over_time.append(agent.esg_impact)
    for asset in assets:
        prices_over_time[asset.name].append(market.prices[assets.index(asset)])

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
