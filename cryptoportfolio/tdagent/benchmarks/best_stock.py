from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np
from collections import deque


class BestStock(TDAgent):
    """ Calculates the asset with the highest return over the whole back-test range and 
        allocates 100% of the capital into this asset until the end of the back-test. 
        All other assets' allocation shares remain constant at 0%, therefore there are 
        no transaction costs throughout the whole back-test. """
    
    def __init__(self, lookback_window_size=1, commission_rate=0.0025):
        self._name = "BestStock"
        super(BestStock, self).__init__(lookback_window_size, commission_rate)
    
    def reset(self, opt_timestep=None):
        """ Resets the environment to its initial state """
        
        if opt_timestep is not None:
            self._timestep = opt_timestep - 1
        else:# Reassigning the initial values
            self._timestep = self._lookback_window_size - 1
           
        # Calculating the asset with the highest return at the end of the back-test
        rel_price_matrix = np.array([self.DataManager.relative_price_vector(n) for n in range(self._crash_length)])
        temp = np.prod(rel_price_matrix, axis=0)[1:] # only non-cash assets
        best_asset = np.argmax(temp)
        initial_weights = np.zeros(rel_price_matrix.shape[1])
        initial_weights[best_asset + 1] = 1 # necessary because I excluded the cash asset, otherwise the index is shifted    
                
        self._PVM = deque([initial_weights], maxlen=2) # weights from the "previous" step
        self._weight_storage = [initial_weights]
        
        # First portfolio value is 1. Just an arbitrary value!
        self._portfolio_values = [1]
        self._final_portfolio_value = 1
        self._rewards = []
        self._rates_of_return = []
        self._sum_of_transaction_costs = 0

        # Return the initial observation
        observation = self.DataManager.price_tensor(self._timestep)
        
        return observation
    
    def decide_weights(self, observation, prev_weights=None):
        weights = self._PVM[-1] # previous weights
        return weights