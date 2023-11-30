from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class EG(TDAgent):
    """ Exponentiated Gradient (EG) algorithm by Helmbold et al.

    Reference:
        Helmbold, David P., et al.
        "On‐Line Portfolio Selection Using Multiplicative Updates."
        Mathematical Finance 8.4 (1998): 325-347.
        http://www.cis.upenn.edu/~mkearns/finread/helmbold98line.pdf
    """

    def __init__(self, eta=0.05, b=None, prev_weights=None):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super(EG, self).__init__()
        self.eta = eta
        self.b = b
        self.prev_weights = prev_weights
        self._name = "EG"

    def init_pw(self, x):
        self.b = np.ones(x.size)

    def decide_weights(self, x, prev_weights):
        self.record_history()
        x = self.history[-1,:].ravel()
        if self.prev_weights is None:
            self.prev_weights = np.ones(x.size) / x.size
        if self.b is None:
            self.init_pw(x)
        else:
            self.b = self.prev_weights * np.exp(self.eta * x.T / np.dot(x,prev_weights))
        b = self.b / np.sum(self.b)
        self.prev_weights = b
        return b