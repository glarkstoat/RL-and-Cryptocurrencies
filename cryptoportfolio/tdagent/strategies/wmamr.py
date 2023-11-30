from cryptoportfolio.tdagent.strategies.pamr import PAMR
import numpy as np


class WMAMR(PAMR):
    """ Weighted Moving Average Passive Aggressive Algorithm for Online Portfolio Selection.
    It is just a combination of OLMAR and PAMR, where we use mean of past returns to predict
    next day's return.

    Reference:
        Li Gao, Weiguo Zhang
        Weighted Moving Averag Passive Aggressive Algorithm for Online Portfolio Selection, 2013.
        http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6643896
    """

    def __init__(self, lookback_window_size=5):
        """
        :param w: Windows length for moving average.
        """
        super(WMAMR, self).__init__(lookback_window_size)
        self._name = "WMAMR"

        if lookback_window_size < 1:
            raise ValueError('window parameter must be >=1')

    def decide_weights(self, x, prev_weights):
        self.record_history()
        xx = np.mean(self.history[-self._lookback_window_size:,], axis=0)
        # calculate return prediction
        b = self.update(prev_weights, xx, self.eps, self.C)

        return b