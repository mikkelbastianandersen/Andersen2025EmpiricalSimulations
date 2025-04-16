import numpy as np

class Market:
    def __init__(self, mu_t, sigma_t, alpha_t, beta_t, rho_t, r ,agents, num_steps, dt=1/252):
        self.dt = dt
        self.mu_t = mu_t
        self.sigma_t = sigma_t
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.rho_t = rho_t
        self.r = r
        self.agents = agents

        self.N = len(mu_t)
        self.K = len(sigma_t)
        self.num_steps = num_steps

        self.esg = np.zeros((self.N,self.num_steps))
        self.prices = np.ones((self.N,self.num_steps))

        self.current_step = 0

    def generate_shocks(self):
        I = np.eye(self.N)
        joint_corr = np.block([
            [I,        self.rho_t],
            [self.rho_t.T,  I     ]
        ])  # shape (2N, 2N)
        L = np.linalg.cholesky(joint_corr)        
        stand_norms = np.random.randn(2 * self.N)
        joint_shocks = L @ stand_norms
        z = joint_shocks[:self.N]
        w = joint_shocks[self.N:]
        return z,w

    def step(self):
        for agent in self.agents:
            agent.decide_portfolio(market_state=self)
            
        # Generate correlated shocks for returns and ESG
        z,w = self.generate_shocks()

        # Compute returns and ESG impacts
        returns = self.mu_t * self.dt + np.sqrt(self.dt) * self.sigma_t @ z
        esg_addition = self.alpha_t * self.dt + np.sqrt(self.dt) * self.beta_t @ w

        # Update the esg and price matrices
        self.esg[:,self.current_step + 1] = self.esg[:,self.current_step] + esg_addition
        self.prices[:,self.current_step + 1] = self.prices[:,self.current_step] * (1 + returns)
        
        # Make sure the price is not negative
        self.prices[:,self.current_step + 1] = np.maximum(self.prices[:,self.current_step + 1], 1e-6)

        for agent in self.agents:
            agent.wealth *= (1 + agent.weights @ returns + (1-sum(agent.weights))*self.r) # Remember risk free
            agent.esg_impact += agent.weights @ esg_addition

        self.current_step += 1

        return self
    def update_beliefs(self):
        # Update beliefs based on the current state of the market
        # For now we keep the beliefs constant, but will be changed later
        self.mu_t = self.mu_t
        self.sigma_t = self.sigma_t
        self.alpha_t = self.alpha_t
        self.beta_t = self.beta_t
        self.rho_t = self.rho_t


