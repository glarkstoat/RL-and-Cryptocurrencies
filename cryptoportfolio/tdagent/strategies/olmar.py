from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class OLMAR(TDAgent):
    """ On-Line Portfolio Selection with Moving Average Reversion
    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    def __init__(self, lookback_window_size=5, commission_rate=0.0025, eps=10, b=None):
        """
        :param window: Lookback window.
        :param eps(epsilon): Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """
        super(OLMAR, self).__init__(lookback_window_size, commission_rate)
        # input check
        if lookback_window_size < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')
        
        self.eps = eps
        self.b = b
        self._name = "OLMAR"
 
    def decide_weights(self, x, prev_weights):
        self.record_history()
        nx = self.DataManager.relative_price_vector(self._timestep)
        
        #predict next price relative vector
        if self.b is None:
            self.b = np.ones(nx.size) / nx.size
        prev_weights = self.b
        if self.history.shape[0] < self._lookback_window_size + 1:
            data_phi=self.history[self.history.shape[0]-1,:]
        else:
            data_phi = np.zeros((1,nx.size))
            tmp_x = np.ones((1,nx.size))
            temp = 1.
            for i in range(self._lookback_window_size):
                data_phi += temp
                tmp_x = np.multiply(tmp_x, self.history[-i-1,:])
                temp = 1. / tmp_x
            data_phi = data_phi * (1./self._lookback_window_size)
        data_phi = np.squeeze(data_phi)
        #update portfolio
        b = self.update(prev_weights, data_phi, self.eps)
        b = b.ravel()
        self.b = b
        return self.b

    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = x.mean()
        ell = max(0, eps - b.dot(x))
        denominator = np.linalg.norm(x-x_mean)**2
        if denominator == 0:
            #zero valatility
            lam = 0
        else:
            lam = ell / denominator
        # update portfolio
        b = b + lam * (x - x_mean)
        # project it onto simplex
        return self.euclidean_proj_simplex(b)