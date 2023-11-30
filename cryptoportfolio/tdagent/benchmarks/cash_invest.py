from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class CASH(TDAgent):
    """ Invests everything into the cash asset. Just for test purposes. I wanna know how the return will look like. """
    
    def __init__(self, lookback_window_size=1, commission_rate=0.0025):
        self._name = "Cash Invest"
        super(CASH, self).__init__(lookback_window_size, commission_rate)
    
    def decide_weights(self, observation, prev_weights=None):
        """ Decides the weights for the next step. Takes the weights that resulted 
        from the price movements in the previous time period without reallocating the 
        capital. """
        
        # unity vector -> everything invested in cash currency
        weights = np.zeros(self._number_of_coins + 1); weights[0] = 1

        return weights
        
