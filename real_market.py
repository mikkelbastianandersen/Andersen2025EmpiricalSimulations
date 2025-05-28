import numpy as np
import yfinance as yf



class RealMarket:
    def __init__(self):
        self.tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META",]

        self.dt = 1 / 252  # Daily time step
        self.n = len(self.tickers)
        
        self.mu, self.sigma = self._calculate_mu_sigma()
        alpha_beta_scale = 100  # Scale factor for alpha and beta
        self.alpha = alpha_beta_scale*np.array([0.049501682, 0.067501, 0.014784, 0.089661, 0.156874])  # Example ESG impact coefficients
        self.beta = alpha_beta_scale*np.array([[0.0000571, -0.0000494, -0.0000016, -0.0001005, -0.0001470], 
                              [-0.0000494, 0.00000857, 0.0000074, 0.0001622, 0.0002847],
                              [-0.0000016,0.0000074,0.000001,0.0000139, 0.0000229],
                              [-0.0001005,0.0001622,0.0000139,0.0003107, 0.0005216],
                              [-0.0001470,0.0002847,0.0000229,0.0005216, 0.0010529]
                              ]) # This data can be found in the data/CO2_emissions.xlsx file.
        
        self.rho = np.eye(self.n) # Correlation matrix

        self.T = 0.5 # Total time in years
        self.S_0 = yf.download(self.tickers, start="2020-01-01", end="2023-12-31")['Close'].iloc[0].values
        self.G_0 = np.zeros(self.n)
        self.r = 0.045 # 10 year US Treasury yield

    
    def _calculate_mu_sigma(self):
        prices = yf.download(self.tickers, start="2020-01-01", end="2023-12-31")['Close']
        daily_log_returns = np.log(prices / prices.shift(1)).dropna()
        annual_log_returns = daily_log_returns.groupby(daily_log_returns.index.year).sum()
        annual_returns = np.exp(annual_log_returns) - 1
        mu = annual_returns.mean().values
        cov = annual_returns.cov().values 
        sigma = np.sqrt(cov)
        return mu, sigma

