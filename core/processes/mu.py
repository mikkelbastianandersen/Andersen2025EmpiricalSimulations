from core.processes.process import ParameterProcess


class MuProcess(ParameterProcess):
    def __init__(self, mu_0):
        self.mu_t = mu_0
        self.t = 0

    def update(self, r_t):
        self.t += 1
        self.mu_t += (r_t - self.mu_t) / self.t

    def get(self):
        return self.mu_t
