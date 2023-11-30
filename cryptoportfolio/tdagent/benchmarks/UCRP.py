from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class UCRP(TDAgent):
    """ Uniform constant rebalancing portfolio strategy. Rebalances the portfolio at every step. """
    
    def __init__(self, lookback_window_size=1, commission_rate=0.0025):
        self._name = "UCRP"
        super(UCRP, self).__init__(lookback_window_size, commission_rate)

    def decide_weights(self, observation, prev_weights=None):
        weights = np.ones(self._number_of_coins + 1) / (self._number_of_coins + 1) # balanced weights
        return weights