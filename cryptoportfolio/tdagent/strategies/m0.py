from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class M0(TDAgent):
    """ Constant rebalanced portfolio = use fixed weights all the time. Uniform weights are commonly used as a benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, beta=0.5, C=None):
        """
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super(M0, self).__init__()
        self.beta = beta
        self.C = C
        self._name = "M0"

    def decide_weights(self, x, prev_weights=None):
        x = self.DataManager.relative_price_vector(self._timestep)
        m = x.size
        if self.C is None:
            self.C = np.zeros((m,1))
        b = (self.C + self.beta) / (m * self.beta + np.ones((1,m)).dot(self.C))
        max_ind = np.argmax(x)
        self.C[max_ind] += 1

        return b.ravel()
