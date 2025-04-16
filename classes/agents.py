class Agent:
    def __init__(self, name, initial_wealth, utility):
        self.name = name
        self.wealth = initial_wealth
        self.esg_impact = 0
        self.utility = utility
        self.weights = None

    def decide_portfolio(self, market_state):
        optimal_weights = self.utility.optimal_portfolio(self, market_state)
        self.weights = optimal_weights

    def update_wealth(self, market):
        returns = market.prices[:,market.current_step + 1]
        esg_scores = market.esg

        portfolio_return = self.portfolio.calculate_return(returns)
        portfolio_esg = self.portfolio.calculate_esg(esg_scores)

        self.wealth *= (1 + portfolio_return)
        self.esg_impact += portfolio_esg