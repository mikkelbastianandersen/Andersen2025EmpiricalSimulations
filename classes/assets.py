# assets.py

class Asset:
    def __init__(self, name, mu, sigma, alpha, beta):
        self.name = name
        self.mu = mu  # drift
        self.sigma = sigma  # volatility
        self.alpha = alpha # drift of esg
        self.beta = beta # volatility of esg

            
