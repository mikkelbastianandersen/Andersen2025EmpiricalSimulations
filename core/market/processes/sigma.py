from core.market.processes.process import ParameterProcess


class SigmaProcess(ParameterProcess):
    def __init__(self, sigma_0):
        self.sigma_t = sigma_0

    def update(self, t):
        pass

    def get(self):
        return self.sigma_t
