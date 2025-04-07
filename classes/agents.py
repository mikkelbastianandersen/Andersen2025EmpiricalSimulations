import numpy as np

class Agent:
    def __init__(self, initial_wealth, risk_aversion, esg_preference, utility):
        self.wealth = initial_wealth
        self.risk_aversion = risk_aversion
        self.esg_preference = esg_preference
        self.utility = utility  # ✅ inject utility class
        self.esg_impact = 0
        self.portfolio = None

    def optimize_portfolio(self, market_state):
        optimal_allocation = self.utility.optimal_portfolio(market_state, self)
        new_allocations = {
            asset_name: weight
            for asset_name, weight in zip(self.portfolio.holdings.keys(), optimal_allocation)
        }
        self.portfolio.update_holdings(new_allocations)

    def update(self, market_state):
        # Update wealth
        portfolio_return = self.portfolio.calculate_value(market_state['returns'])
        self.wealth *= (1 + portfolio_return)

        # Update ESG impact
        portfolio_esg = self.portfolio.esg_impact(market_state['esg'])
        self.esg_impact += portfolio_esg

