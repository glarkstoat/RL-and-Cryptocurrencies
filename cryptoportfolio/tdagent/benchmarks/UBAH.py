from cryptoportfolio.tdagent.tdagent import TDAgent


class UBAH(TDAgent):
    """ Uniform Buy And Hold strategy. Basically does not reallocate the weights. Just leaves the portfolio 
    alone and waits for the result, while letting the weights shift naturally due to the price movements. """
    
    def __init__(self, lookback_window_size=1, commission_rate=0.0025):
        self._name = "UBAH"
        super(UBAH, self).__init__(lookback_window_size, commission_rate)

    def decide_weights(self, observation, prev_weights=None):
        """ Decides the weights for the next step. Takes the weights that resulted 
        from the price movements in the previous time period without reallocating the 
        capital. """
        
        y_t = self.DataManager.relative_price_vector(self._timestep)
        w_t_1 = self._PVM[-1]
        weights = (y_t * w_t_1) / (y_t @ w_t_1) # w_prime from formula (7) from (A deep RL Framework for ...)

        return weights