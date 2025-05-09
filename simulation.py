from core.processes.alpha import AlphaProcess
from core.processes.beta import BetaProcess
from core.processes.mu import MuProcess
from core.processes.sigma import SigmaProcess
from core.asset import RiskyAssets
from core.esg import ESGImpacts
from core.noise import CorrelatedNoise
from core.data import DataCollector
from config import mu_0, sigma_0, alpha_0, beta_0, rho_0, S0, G0, dt, T, n_assets


def run_simulation():
    # Initialize processes
    mu = MuProcess(mu_0)
    sigma = SigmaProcess(sigma_0)
    alpha = AlphaProcess(alpha_0)
    beta = BetaProcess(beta_0)

    # Initialize models
    risky_asset = RiskyAssets(S0)
    esg_impact = ESGImpacts(G0)
    collector = DataCollector(T, n_assets)

    collector.prices[0] = S0
    collector.esg[0] = G0

    for t in range(T):
        # Get current parameters
        mu_t = mu.get()
        sigma_t = sigma.get()
        alpha_t = alpha.get()
        beta_t = beta.get()

        # Update correlated noise matrix (if sigma/beta/rho evolve)
        correlated_noise = CorrelatedNoise(sigma_t, beta_t, rho_0)

        # Sample noise
        dz, dw = correlated_noise.sample(dt)

        # Step forward dynamics
        r_t, S_t = risky_asset.step(dt, dz, mu_t, sigma_t)
        G_t = esg_impact.step(dt, dw, alpha_t, beta_t)

        # Update parameter processes
        mu.update(r_t)
        sigma.update(t)
        alpha.update(t)
        beta.update(t)

        collector.record(r_t, S_t, G_t, mu_t, sigma_t, beta_t, rho_0)

    return collector
