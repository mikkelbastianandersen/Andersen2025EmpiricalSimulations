import numpy as np

class Agent:
    def __init__(self, initial_wealth, risk_aversion, esg_preference):
        self.wealth = initial_wealth
        self.risk_aversion = risk_aversion  # a_j
        self.esg_preference = esg_preference  # d_j
        self.esg_impact = 0
        self.portfolio = None  # will be assigned later

    def optimize_portfolio(self, market_state):
        # --- Agent parameters ---
        a_j = self.risk_aversion
        d_j = self.esg_preference
        wealth = self.wealth

        # --- Market parameters ---
        mu = market_state['mu']            # (n_assets,)
        alpha = market_state['alpha_esg']  # (n_assets,)
        r = market_state['risk_free_rate']

        Sigma_W = market_state['cov_returns']   # (n_assets, n_assets)
        Sigma_Z = market_state['cov_esg']       # (n_assets, n_assets)
        Sigma_WZ = market_state['cov_cross']    # (n_assets, n_assets)

        # --- Compute M matrix ---
        M = (a_j ** 2) * Sigma_W + (d_j ** 2) * Sigma_Z + 2 * a_j * d_j * Sigma_WZ

        # --- Compute the target vector ---
        excess_return = mu - r
        target_vector = a_j * excess_return + d_j * alpha

        # --- Solve for optimal allocation ---
        try:
            optimal_allocation = np.linalg.solve(M, target_vector) / wealth
        except np.linalg.LinAlgError:
            # If M is singular, fallback (you can log or raise error if desired)
            optimal_allocation = np.zeros_like(mu)

        # --- Update portfolio holdings ---
        new_allocations = {
            asset_name: weight for asset_name, weight in zip(self.portfolio.holdings.keys(), optimal_allocation)
        }
        self.portfolio.update_holdings(new_allocations)

    def update(self, market_state):
        # Update wealth
        portfolio_return = self.portfolio.calculate_value(market_state['returns'])
        self.wealth *= (1 + portfolio_return)

        # Update ESG impact
        portfolio_esg = self.portfolio.esg_impact(market_state['esg'])
        self.esg_impact += portfolio_esg

