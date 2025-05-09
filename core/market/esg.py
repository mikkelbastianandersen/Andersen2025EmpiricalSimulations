import numpy as np
from config import T, n_assets


class ESGImpacts:
    def __init__(self, G0):
        self.G = np.zeros((T + 1, n_assets))
        self.G[0] = G0.copy()
        self.t = 0

    def step(self, dt, dw, alpha_t, beta_t):
        G_prev = self.G[self.t]
        G_new = G_prev + alpha_t * dt + beta_t @ dw
        self.t += 1
        self.G[self.t] = G_new
        return G_new
