from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class SP(TDAgent):
    '''Switch Portfolio'''
    def __init__(self, gamma=0.25, prev_weights=None):
        super(SP, self).__init__()
        self.gamma = gamma
        self._prev_weights = prev_weights
        self._name = "SP"

    def decide_weights(self, x, prev_weights=None):
        self.record_history()
        nx = self.history[-1,:].ravel()
        if self._prev_weights is None:
            self._prev_weights = np.ones(nx.size) / nx.size
        b = self._prev_weights * (1-self.gamma-self.gamma/nx.size) + self.gamma/nx.size
        b = b / np.sum(b)
        self._prev_weights = b
        return b