from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class OLMAR2(TDAgent):
    '''Moving average reversion strategy for on-line portfolio selection
    Reference:
        Bin Li, Steven C.H. Hoi, Doyen Sahoo, Zhi-Yong Liu
    '''

    def __init__(self,  eps=10, alpha=0.5, data_phi=None, b=None):
        '''init
        :param eps: mean reversion threshold
        :param alpha: trade off parameter for moving average
        '''
        super(OLMAR2, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.data_phi = data_phi
        self.b = b
        self._name = "OLMAR2"
  
    def decide_weights(self, x, prev_weights):
        self.record_history()
        nx = self.DataManager.relative_price_vector(self._timestep)

        if self.b is None:
            self.b = np.ones(nx.size) / nx.size
        prev_weights = self.b
        if self.data_phi is None:
            self.data_phi = np.ones((1,nx.size))
        else:
            self.data_phi = self.alpha + (1-self.alpha)*self.data_phi/nx

        ell = max(0, self.eps - self.data_phi.dot(prev_weights))

        x_bar = self.data_phi.mean()
        denominator = np.linalg.norm(self.data_phi - x_bar)**2

        if denominator == 0:
            lam = 0
        else:
            lam = ell / denominator

        self.data_phi = np.squeeze(self.data_phi)
        b = prev_weights + lam * (self.data_phi - x_bar)

        b = self.euclidean_proj_simplex(b)
        self.b = b
        return self.b