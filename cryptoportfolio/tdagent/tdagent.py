import gym
from gym import spaces
import numpy as np
from collections import deque 
import logging
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, euclidean
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from numba import jit

from cryptoportfolio.data.datamanager import DataManager
from cryptoportfolio.tools.performance_measures import sharpe_ratio, max_drawdown
from cryptoportfolio.tools.trading_utils import calculate_reward, transaction_factor, portfolio_value
from cryptoportfolio.tools.trading_utils import plot_portfolio_values, plot_weight_changes, plot_rewards


class TDAgent(gym.Env):
    
    def __init__(
        self, 
        lookback_window_size : int = 1,
        commission_rate : float = 0.0025
    ):
        super(TDAgent, self).__init__()

        self._lookback_window_size = lookback_window_size
        self._features = ["close", "low", "high"]
        self._feature_number = len(self._features)        
        self._commission_rate = commission_rate
        self._PVM = deque(maxlen=2)
        self.history = None
        
        # Initiate DataManager that manages the portfolio
        self.DataManager = DataManager(self._lookback_window_size, self._features)
        self._number_of_coins = 12
        
        # Action space: the portfolio weights for the coins in the portfolio plus
        # the cash currency e.g. Tether
        self.action_space = spaces.Box(low=0, high=1, shape=(self._number_of_coins + 1, # includes the weight of the cash currency
                                                             ), 
                                       dtype=np.float32)
        
        # State space: the price tensor
        # low and high values so that there's no warning from gym.check_env but ARBITRARY VALUES FOR NOW!
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._feature_number,
                                                                   self._number_of_coins, 
                                                                   self._lookback_window_size), 
                                            dtype=np.float32)
    
    def reset(self, opt_timestep=None):
        """ Resets the environment to its initial state """
        
        if opt_timestep is not None:
            self._timestep = opt_timestep - 1
        else:# Reassigning the initial values
            self._timestep = self._lookback_window_size - 1
           
        initial_weights = np.ones(self._number_of_coins + 1) / (self._number_of_coins + 1) # balanced weights
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
    
    def step(self, action):
        """ The result from taking a specific action. Calculates the reward and 
        gets the next state. """
        
        # Adding the new portfolio weights to the PVM
        self._PVM.append(action)
        self._weight_storage.append(action)
        
        # Get new values
        y_t = self.DataManager.relative_price_vector(self._timestep)
        mu_t, sum_term = transaction_factor(y_t, self._PVM, self._commission_rate)

        # Calculating the reward using the newly calculated portfolio weights
        reward = calculate_reward(y_t, self._PVM[-2], mu_t)
        self.update_performance_measures(reward, y_t, mu_t, sum_term)
            
        reward /= self._crash_length # immediate reward from formula (22) in (A deep learning framework for ...)
        self._rewards.append(reward)
        
        # Returning the info
        info = {"PVM":self._PVM, "reward":reward, 
                "transaction_factor":mu_t}

        # Terminating if agent has reached the end of the crash        
        if self._timestep == self._crash_length-1:
            done = True
            observation = self.DataManager.price_tensor(self._timestep)
            return observation, reward, done, info
        else:
            done = False
            
        # Returning the new state
        self._timestep += 1
        observation = self.DataManager.price_tensor(self._timestep)
        
        return observation, reward, done, info
    
    def render(self):
        plot_portfolio_values(self._portfolio_values)
        plot_weight_changes(self._weight_storage, self._number_of_coins, self.DataManager._portfolio.keys())
        plot_rewards(self._rewards)
        
        print("Return:", np.sum(self._rewards))
        print("Final Portfolio Value: ", self._final_portfolio_value)
        print("Sharpe Ratio:", sharpe_ratio(self._rates_of_return))
        print("Max Drawdown:", max_drawdown(self._portfolio_values))
        print(f"Commission rate {self._commission_rate}")
        print(f"Sum of transaction costs {self._sum_of_transaction_costs}")
    
    def close (self):
        pass

    def generate_portfolio(self, year="2021", synthetic=False, recycle=True, split="whole"):
        self._crash_length = self.DataManager.generate_portfolio(year=year, synthetic=synthetic, recycle=recycle, split=split)
        self._number_of_coins = self.DataManager._number_of_coins
        
    def update_performance_measures(self, reward, y_t, mu_t, sum_term):
        """ Updates the final portfolio value, the rates of return and the portfolio values """
            
        temp = np.exp(reward) # mu_t * y_t * w_t-1 # exp to get get rid of the ln
        
        # Calculating the performance measures
        self._final_portfolio_value *= temp # from formula (11) in (A deep learning framework for ...)
        
        rate_of_return = temp - 1 # from formula (9) in (A deep learning framework for ...)
        self._rates_of_return.append(rate_of_return)
        
        new_portfolio_value = portfolio_value(self._portfolio_values[-1], mu_t, y_t, self._PVM[-2])
        self._portfolio_values.append(new_portfolio_value)
        #print("Portfolio values[-2], [-1]:")
        #print(self._portfolio_values[-2], self._portfolio_values[-1])
        
        self._sum_of_transaction_costs += sum_term
        
    def performance_measures(self):
        dictionary = {"return" : round(np.sum(self._rewards), 5),
               "fPV" : round(self._final_portfolio_value, 5),
               "sharpe" : round(sharpe_ratio(self._rates_of_return), 5), 
               "mdd" : round(max_drawdown(self._portfolio_values), 5),
               "commission_rate" : round(self._commission_rate, 5),
               "sum_of_transaction_costs" : round(self._sum_of_transaction_costs, 5)}
        return dictionary
    
    #@jit(nopython=True, fastmath=True)
    def simplex_proj(self, y):
        '''projection of y onto simplex. '''
        m = len(y)
        bget = False

        s = sorted(y, reverse = True)
        tmpsum = 0.

        for ii in range(m-1):
            tmpsum = tmpsum + s[ii]
            tmax = (tmpsum - 1) / (ii + 1)
            if tmax >= s[ii+1]:
                bget = True
                break
        
        if not bget:
            tmax = (tmpsum + s[m-1] - 1) / m

        return np.maximum(0, y-tmax)

    def record_history(self):
        nx = self.DataManager.relative_price_vector(self._timestep)
        nx = np.reshape(nx, (1,nx.size))
        if self.history is None:
            self.history = nx
        else:
            self.history = np.vstack((self.history, nx))
            
    def euclidean_proj_simplex(self, v, s=1):
        '''Compute the Euclidean projection on a positive simplex
        :param v: n-dimensional vector to project
        :param s: int, radius of the simple
        return w numpy array, Euclidean projection of v on the simplex
        Original author: John Duchi
        '''
        assert s>0, "Radius s must be positive (%d <= 0)" % s

        n, = v.shape # raise ValueError if v is not 1D
        # check if already on the simplex
        if v.sum() == s and np.alltrue( v>= 0):
            return v

        # get the array of cumulaive sums of a sorted copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of >0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.)
        w = (v-theta).clip(min=0)
        return w
    
    #@jit(nopython=True, fastmath=True)
    def mc_simplex(self, d, points):
        '''Sample random points from a simplex with dimension d
        :param d: Number of dimensions
        :param points: Total number of points.
        '''
        a = np.sort(np.random.random((points,d)))
        a = np.hstack([np.zeros((points,1)), a, np.ones((points,1))])
        return np.diff(a)
    
    #@jit(nopython=True, fastmath=True)
    def l1_median_VaZh(self, X, eps=1e-5):
        '''calculate the L1_median of X with the l1median_VaZh method
        '''
        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)
            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r==0 else num_zeros/r
                y1 = max(0, 1-rinv)*T + min(1, rinv)*y

            if euclidean(y, y1) < eps:
                return y1

            y = y1
            
    def get_close(self):
        '''get close data from relative price
        :param x: relative price data
        '''
        close = np.ones(self.history.shape)
        for i in range(1,self.history.shape[0]):
            close[i,:] = close[i-1] * self.history[i,:]
        return close