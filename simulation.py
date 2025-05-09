# Processes
from core.market.processes.alpha import AlphaProcess
from core.market.processes.beta import BetaProcess
from core.market.processes.mu import MuProcess
from core.market.processes.sigma import SigmaProcess
from core.market.processes.rho import RhoProcess

# Models
from core.market.asset import RiskyAssets
from core.market.esg import ESGImpacts

# Helpers
from core.market.noise import CorrelatedNoise
from core.data import DataCollector


class Simulation:
    def __init__(self, config):

        self.T = config.T
        self.dt = config.dt

        # Initialize processes
        self.mu = MuProcess(config.mu_0)
        self.sigma = SigmaProcess(config.sigma_0)
        self.alpha = AlphaProcess(config.alpha_0)
        self.beta = BetaProcess(config.beta_0)
        self.rho = RhoProcess(config.rho_0)

        # Initialize models
        self.asset = RiskyAssets(config.S0)
        self.esg = ESGImpacts(config.G0)
        self.collector = DataCollector(self.T, config.n_assets)
        self.collector.prices[0] = config.S0
        self.collector.esg[0] = config.G0

    def step(self, t):
        mu_t = self.mu.get()
        sigma_t = self.sigma.get()
        alpha_t = self.alpha.get()
        beta_t = self.beta.get()
        rho_t = self.rho.get()

        correlated_noise = CorrelatedNoise(sigma_t, beta_t, rho_t)
        dz, dw = correlated_noise.sample(self.dt)

        r_t, S_t = self.asset.step(self.dt, dz, mu_t, sigma_t)
        G_t = self.esg.step(self.dt, dw, alpha_t, beta_t)

        self.collector.record(r_t, S_t, G_t, mu_t, sigma_t, alpha_t, beta_t, rho_t)

        self.mu.update(r_t)
        self.sigma.update(t)
        self.alpha.update(t)
        self.beta.update(t)
        self.rho.update(t)

    def run(self):
        for t in range(self.T):
            self.step(t)
        return self.collector
