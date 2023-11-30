from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np


class RandomAgent(TDAgent):
    """ Always chooses random weights """
    
    def __init__(self, lookback_window_size=1, commission_rate=0.0025):
        self._name = "RandomAgent"
        super(RandomAgent, self).__init__(lookback_window_size, commission_rate)
    
    def decide_weights(self, observation, prev_weights=None):
        # Random weights that sum to 1
        weights = self.action_space.sample()
        weights /= np.sum(weights)
        return weights