from core.market.processes.process import ParameterProcess


class RhoProcess(ParameterProcess):
    def __init__(self, rho_0):
        self.rho_t = rho_0

    def update(self, t):
        pass

    def get(self):
        return self.rho_t
    