from .noise import CorrelatedNoise


class Market:
    def __init__(self, risk_free, risky, esg, rho):
        self.risk_free = risk_free
        self.risky = risky
        self.esg = esg
        self.rho = rho

    def step(self, dt):
        noise = CorrelatedNoise(self.risky.sigma_t, self.esg.beta_t, self.rho)
        dz, dw = noise.sample(dt)

        B = self.risk_free.step(dt)
        S = self.risky.step(dt, dz)
        G = self.esg.step(dt, dw)
        return B, S, G
