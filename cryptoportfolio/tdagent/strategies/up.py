from cryptoportfolio.tdagent.tdagent import TDAgent

import numpy as np
from collections import deque 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class UP(TDAgent):
    """ Universal Portfolio by Thomas Cover enhanced for "leverage" (instead of just
        taking weights from a simplex, leverage allows us to stretch simplex to
        contain negative positions).
    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """
    def __init__(self, eval_points=10000, leverage=1., W=None):
        """
        :param eval_points: Number of evaluated points (approximately). Complexity of the
            algorithm is O(time * eval_points * nr_assets**2) because of matrix multiplication.
        :param leverage: Maximum leverage used. leverage == 1 corresponds to simplex,
            leverage == 1/nr_stocks to uniform CRP. leverage > 1 allows negative weights
            in portfolio.
        """
        super(UP, self).__init__()
        self._name = "UP"
        self.eval_points = eval_points
        self.leverage = leverage
        self.W = W
    
    def init_portfolio(self, X):
        """ Create a mesh on simplex and keep wealth of all strategies. """
        m = X.shape[1]
        # create set of CRPs
        self.W = np.matrix(self.mc_simplex(m - 1, self.eval_points))
        self.S = np.matrix(np.ones(self.W.shape[0])).T

        # stretch simplex based on leverage (simple calculation yields this)
        leverage = max(self.leverage, 1./m)
        stretch = (leverage - 1./m) / (1. - 1./m)
        self.W = (self.W - 1./m) * stretch + 1./m

    def decide_weights(self, observation, prev_weights=None):
        # calculate new wealth of all CRPs
        x = self.DataManager.relative_price_vector(self._timestep)
        x = np.reshape(x, (1,x.size))

        if self.W is None:
            self.init_portfolio(x)

        self.S = np.multiply(self.S, self.W * np.matrix(x).T)
        b = self.W.T * self.S
        pv = b / np.sum(b)
        pvn = np.ravel(pv)
        return pvn
