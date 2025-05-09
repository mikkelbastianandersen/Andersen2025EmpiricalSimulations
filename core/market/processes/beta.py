from core.market.processes.process import ParameterProcess


class BetaProcess(ParameterProcess):
    def __init__(self, beta_0):
        self.beta_t = beta_0

    def update(self, t):
        # e.g., random walk beta
        pass

    def get(self):
        return self.beta_t
