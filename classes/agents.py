class Agent:
    def __init__(self, name, initial_wealth, risk_aversion, esg_preference, utility):
        self.name = name
        self.wealth = initial_wealth
        self.risk_aversion = risk_aversion
        self.esg_preference = esg_preference
        self.utility = utility
        self.portfolio = None
        self.esg_impact = 0

    def decide_portfolio(self, market_state):
        optimal_weights = self.utility.optimal_portfolio(self, market_state)
        self.portfolio.update_holdings(optimal_weights)

    def update(self, market_state):
        returns = market_state['returns']
        esg_scores = market_state['esg']

        portfolio_return = self.portfolio.calculate_return(returns)
        portfolio_esg = self.portfolio.calculate_esg(esg_scores)

        self.wealth *= (1 + portfolio_return)
        self.esg_impact += portfolio_esg