


class Agent:
    def __init__(self, utility: object, initial_wealth, initial_esg=0.0):
        self.utility = utility
        self.wealth = initial_wealth
        self.esg = initial_esg
        self.portfolio = None

    def decide_portfolio(self, mu, sigma, alpha, beta, rho, r):
        self.portfolio = self.utility.optimal_portfolio(mu, sigma, alpha, beta, rho, r, self.wealth)

    def update(self, r_t, G_t):
        self.esg += G_t @ self.portfolio * self.wealth
        self.wealth *= (1 + r_t @ self.portfolio)
