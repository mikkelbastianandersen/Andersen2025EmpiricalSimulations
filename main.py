from simple_market import SimpleMarket
from utility import Utility, ExponentialUtility, CobbDouglasUtility
from agent_in_simple_market import AgentInSimpleMarket
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import random
import pandas as pd
from real_market import RealMarket

np.random.seed(42)  # For reproducibility


if __name__ == "__main__":
    # # Simple market parameters
    
    # S_0 = np.array([100, 100])  # Initial prices of the assets
    # G_0 = np.array([100, 100])  # Initial ESG impact
    # mu = np.array([0.07, 0.05])  # Expected returns
    # sigma = np.array([[0.25, 0.05], [0.05, 0.15]])  # Volatility matrix
    # alpha = np.array([0.05, 0.03])  # Expected ESG impact
    # beta = np.array([[0.10, 0.02], [0.02, 0.08]])  # Volatility matrix for ESG impact
    # rho = np.array([[0.2, 0.1], [0.0, 0.3]])  # Correlation matrix
    # r = 0.02  # Risk-free rate
    # T = 1.0   # Total time in years
    # dt = 1/252  # Time step in years (daily)

    # market = SimpleMarket(S_0, G_0, mu, sigma, alpha, beta, rho, r, T, dt)

    # # Plot the results
    #     # Plotting the results. Create one line for each asset's price over time and same for ESG impact.
    # plt.figure(figsize=(12, 6))
    # for i in range(market.S.shape[0]):
    #     plt.plot(market.S[i, :], label=f'Asset {i+1} Price')
    # plt.title('Asset Prices Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # for i in range(market.G.shape[0]):
    #     plt.plot(market.G[i, :], label=f'Asset {i+1} ESG impact')
    # plt.title('Asset ESG impacts Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('ESG Impact')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # agent1 = AgentInSimpleMarket(
    #     utility=CobbDouglasUtility(a=0.6),
    #     market=market,
    #     initial_wealth=100,
    #     initial_esg_impact=100
    # )
    # plt.figure(figsize=(12, 6))
    # plt.plot(agent1.X.T, label='Wealth')
    # plt.title('Agent Wealth Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Wealth')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(agent1.Y.T, label='ESG Impact')
    # plt.title('Agent ESG Impact Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('ESG Impact')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Real market parameters
    real_market = RealMarket() # Contains estimates for mu, sigma, alpha, beta, rho for 5 assets: "APPL", "MSFT", "AMZN", "GOOGL", "META"
    # Using the estimated parameters from the real market to create a SimpleMarket instance
    # Note: we do not use the real market's returns but rather the simulated returns from the SimpleMarket class.
    real_market_sim = SimpleMarket(real_market.S_0, real_market.G_0, real_market.mu, real_market.sigma, real_market.alpha, real_market.beta, real_market.rho, real_market.r, real_market.T, real_market.dt)

    plt.figure(figsize=(12, 6))
    for i in range(real_market_sim.S.shape[0]):
        plt.plot(real_market_sim.S[i, :], label=f'Asset {i+1} Price')
    plt.title('Asset Prices Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    for i in range(real_market_sim.G.shape[0]):
        plt.plot(real_market_sim.G[i, :], label=f'Asset {i+1} ESG impact')
    plt.title('Asset ESG impacts Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('ESG Impact')
    plt.legend()
    plt.grid()
    plt.show()

    print("Simulated market data for real market assets:")

    agent2 = AgentInSimpleMarket(
        utility=ExponentialUtility(a=1, b=1),
        market=real_market_sim,
        initial_wealth=100,
        initial_esg_impact=0
    )
    agent3 = AgentInSimpleMarket(
        utility=ExponentialUtility(a=1, b=0),
        market=real_market_sim,
        initial_wealth=100,
        initial_esg_impact=0
    )

    plt.figure(figsize=(12, 6))
    plt.plot(agent2.X.T, label='Wealth')
    plt.title('Agent Wealth Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Wealth')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(agent2.Y.T, label='ESG Impact')
    plt.title('Agent ESG Impact Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('ESG Impact')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(-agent2.utility_ts, label='a=1, b=1')
    plt.plot(-agent3.utility_ts, label='a=1, b=0')
    plt.title('Agent Utility Over Time - Exponential Utility')
    plt.xlabel('Time Steps')
    plt.ylabel('ax+by')
    plt.legend()
    plt.grid()
    plt.show()

    # agent4 = AgentInSimpleMarket(
    #     utility=CobbDouglasUtility(a=0.5),
    #     market=real_market_sim,
    #     initial_wealth=100,
    #     initial_esg_impact=100
    # )
    # agent5 = AgentInSimpleMarket(
    #     utility=CobbDouglasUtility(a=1),
    #     market=real_market_sim,
    #     initial_wealth=100,
    #     initial_esg_impact=100
    # )

    # plt.figure(figsize=(12, 6))
    # plt.plot(agent4.utility_ts, label='a=0.5')
    # plt.plot(agent5.utility_ts, label='a=1')
    # plt.title('Agent Utility Over Time - Cobb Douglas Utility')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Utility')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Now simulate 1000 real markets and plot the average path of utility for each agent type
    num_simulations = 1000
    utility_exponential_a1_b1 = []
    utility_exponential_a1_b0 = []
    for _ in range(num_simulations):
        real_market_sim = SimpleMarket(real_market.S_0, real_market.G_0, real_market.mu, real_market.sigma, real_market.alpha, real_market.beta, real_market.rho, real_market.r, real_market.T, real_market.dt)
        
        agent6 = AgentInSimpleMarket(
            utility=ExponentialUtility(a=1, b=1),
            market=real_market_sim,
            initial_wealth=100,
            initial_esg_impact=0
        )
        agent7 = AgentInSimpleMarket(
            utility=ExponentialUtility(a=1, b=0),
            market=real_market_sim,
            initial_wealth=100,
            initial_esg_impact=0
        )

        utility_exponential_a1_b1.append(agent6.utility_ts)
        utility_exponential_a1_b0.append(agent7.utility_ts)
    
    average_utility_exponential_a1_b1 = np.mean(utility_exponential_a1_b1, axis=0)
    average_utility_exponential_a1_b0 = np.mean(utility_exponential_a1_b0, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(-average_utility_exponential_a1_b0, label='a=1, b=0')
    plt.plot(-average_utility_exponential_a1_b1, label='a=1, b=1')
    plt.title('Agent Utility Over Time - Exponential Utility')
    plt.xlabel('Time Steps')
    plt.ylabel('ax+by')
    plt.legend()
    plt.grid()
    plt.show()





