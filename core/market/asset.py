import numpy as np
from config import n_assets, T


class RiskyAssets:
    def __init__(self, S0):
        self.S = np.zeros((T + 1, n_assets))
        self.S[0] = S0.copy()
        self.t = 0  # time index

    def step(self, dt, dz, mu_t, sigma_t):
        S_prev = self.S[self.t]
        drift = (mu_t - 0.5 * np.diag(sigma_t @ sigma_t.T)) * dt
        diffusion = sigma_t @ dz
        log_return = drift + diffusion
        S_new = S_prev * np.exp(log_return)
        self.t += 1
        self.S[self.t] = S_new
        return log_return, S_new

    def get_prices(self):
        return self.S[: self.t + 1]  # returns prices up to current time


class RiskFreeAsset:
    def __init__(self, B0, r):
        self.B = [B0]
        self.r = r

    def step(self, dt):
        B_prev = self.B[-1]
        B_new = B_prev * np.exp(self.r * dt)
        self.B.append(B_new)
        return B_new
