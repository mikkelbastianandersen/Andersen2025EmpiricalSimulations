from core.market.processes.process import ParameterProcess


class AlphaProcess(ParameterProcess):
    def __init__(self, alpha_0):
        self.alpha_t = alpha_0

    def update(self, t):
        # e.g., mean-reverting or time-varying alpha
        pass

    def get(self):
        return self.alpha_t
