'''
This agent was set up based on the paper (A Deep Reinforcement Learning Framework for the
Financial Portfolio Management Problem). 
All formulas that are referenced with parentheses are the respective formulas in the paper. 
E.g. formula (7)
'''

import gym
from gym import spaces
import numpy as np
import torch
from collections import deque
import torch.nn.functional as F
import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from cryptoportfolio.data.datamanager import DataManager
from cryptoportfolio.tools.performance_measures import sharpe_ratio, max_drawdown
from cryptoportfolio.tools.trading_utils import calculate_reward, transaction_factor, portfolio_value
from cryptoportfolio.tools.trading_utils import plot_portfolio_values, plot_weight_changes, plot_rewards, plot_transaction_costs


class RLAgent(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(
        self, 
        lookback_window_size: int = 50, 
        commission_rate: float = 0.0025, 
        features: list = ["close", "low", "high"],
        batch_size: int = 10,
    ):
        super(RLAgent, self).__init__()       
        
        self._lookback_window_size = lookback_window_size
        self._features = features
        self._feature_number = len(self._features)
        self._commission_rate = commission_rate
        self._batch_size = batch_size
        self._name = "PPO"
        
        ######
        #self.runtime_step_phase1 = 0
        #self.runtime_step_perf_measures = 0
        #self.runtime_step_obs_tensor = 0
        ######
        
        
        # Initiate DataManager that manages the portfolio
        self.DataManager = DataManager(self._lookback_window_size, self._features)
        self._number_of_coins = 12
        self._weight_storage = [self.initial_weights()]
        
        # Action space: the portfolio weights for the coins in the portfolio plus the cash currency e.g. Tether
        self.action_space = spaces.Box(low=0, high=1, shape=(self._number_of_coins + 1,
                                                             ), 
                                       dtype=np.float32)
        
        # State space: the price tensor low and high values so that there's no warning from gym.check_env 
        # but ARBITRARY VALUES FOR NOW!
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._feature_number,
                                                                   self._number_of_coins, 
                                                                   self._lookback_window_size), 
                                            dtype=np.float32)

    def reset(self):
        """ Resets the environment to its initial state """

        # Reassigning the initial values
        self._timestep = self._lookback_window_size - 1
        
        # Weights from the previous step
        self._PVM = deque([self.initial_weights()], maxlen=2)
        # Contains the indices of the randomly selected batches that correspond to the weights in weight_storage
        self._indices = deque([], maxlen=1)

        # First portfolio value is 1. Just an arbitrary value!
        self._portfolio_values = [1]
        self._final_portfolio_value = 1
        self._rewards = []
        self._rates_of_return = []
        self._sum_of_transaction_costs = 0
        self._transaction_costs = [0]

        # Return the initial observation
        observation = self.DataManager.price_tensor(self._timestep)

        return observation
    
    def step(self, action):
        """ The result from taking a specific action. Calculates the reward and 
        gets the next state. """

        ###########
        #start = datetime.datetime.now()
        
        # Applying the softmax function manually
        action = F.softmax(torch.from_numpy(action), dim=0).numpy()
        self.check_sum(action)        

        # Adding the new portfolio weights to the PVM
        self._PVM.append(action)
        self._weight_storage.append(action)
        
        #runtime = (datetime.datetime.now() - start).total_seconds()
        #self.runtime_step_phase1 += runtime
        ############
        
        ###########
        #start = datetime.datetime.now()
        
        # Get new values
        y_t = self.DataManager.relative_price_vector(self._timestep)
        mu_t, sum_term = transaction_factor(y_t, self._PVM, self._commission_rate)
                
        # Calculating the reward using the newly calculated portfolio weights
        reward = calculate_reward(y_t, self._PVM[-2], mu_t)
        self.update_performance_measures(reward, y_t, mu_t, sum_term)
        
        # Immediate reward from formula (22) in (A deep learning framework for ...)
        reward /= self._crash_length 
        self._rewards.append(reward)
        
        #runtime = (datetime.datetime.now() - start).total_seconds()
        #self.runtime_step_perf_measures += runtime
        ###################
        
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
        
        ################
        #start = datetime.datetime.now()
        
        # Returning the new state
        self._timestep += 1
        observation = self.DataManager.price_tensor(self._timestep)
        
        #runtime = (datetime.datetime.now() - start).total_seconds()
        #self.runtime_step_obs_tensor += runtime
        #################
        
        return observation, reward, done, info
    
    def check_sum(self, actions):
        # Checks if the actions are is normalized correctly
        action_sum = np.sum(actions)
        assert action_sum > 0.9999 and action_sum < 1.0001, (
            f"Action is not normalized. " 
            f"Sum: {action_sum}"
            f"Actions: {actions}"
        )
    
    def render(self):
        plot_portfolio_values(self._portfolio_values)
        plot_weight_changes(self._weight_storage, self._number_of_coins, self.DataManager._portfolio._coins)
        #plot_rewards(self._rewards)
        plot_transaction_costs(self._transaction_costs)

        print("Return:", np.sum(self._rewards))
        print("Final Portfolio Value: ", self._final_portfolio_value, end="\n")
        print("Sharpe Ratio:", sharpe_ratio(self._rates_of_return), end="\n")
        print("Max Drawdown:", max_drawdown(self._portfolio_values), end="\n")
        print(f"Commission rate {self._commission_rate}", end="\n")
        print(f"Sum of transaction costs {self._sum_of_transaction_costs}", end="\n")
        
    def close (self):
        pass
    
    def performance_measures(self):
        dictionary = {"return" : round(np.sum(self._rewards), 9),
               "fPV" : round(self._final_portfolio_value, 9),
               "sharpe" : round(sharpe_ratio(self._rates_of_return), 9), 
               "mdd" : round(max_drawdown(self._portfolio_values), 9),
               "commission_rate" : round(self._commission_rate, 9),
               "sum_of_transaction_costs" : round(self._sum_of_transaction_costs, 9)}
        return dictionary    
    
    def update_performance_measures(self, reward, y_t, mu_t, sum_term):
        """ Updates the final portfolio value, the rates of return and the portfolio values """
        
        temp = np.exp(reward)
        self._final_portfolio_value *= temp
        rate_of_return = temp - 1
        self._rates_of_return.append(rate_of_return)
        new_portfolio_value = portfolio_value(self._portfolio_values[-1], mu_t, y_t, self._PVM[-2])
        self._portfolio_values.append(new_portfolio_value)
        self._sum_of_transaction_costs += sum_term
        self._transaction_costs.append(self._sum_of_transaction_costs)
        
    def generate_portfolio(self, year="2021", synthetic=False, recycle=True, overwrite=False, split="whole"):
        self._crash_length = self.DataManager.generate_portfolio(year=year, synthetic=synthetic, recycle=recycle, split=split)
        self._number_of_coins = self.DataManager._number_of_coins
    
        if overwrite:
            assert synthetic == True, "Can't overwrite the real crash portfolio. Pass synthetic=True."
            self.overwrite_synthetic_data(recycle, overwrite, year)
    
    def get_random_weights(self):
        
        random_weights = self.action_space.sample()
        while sum(random_weights) == 0:
            random_weights = self.action_space.sample()
        
        return random_weights
    
    def normalize_weights(self, weights):
        
        if sum(weights) != 1.0:
            weights /= sum(weights)
            return weights
        else:
            return weights
    
    def balanced_weights(self):
        """ Returns uniformely balanced weights were the capital is equally distributed 
        across the entire portfolio """
        return np.ones(self._number_of_coins) / self._number_of_coins
    
    def custom_reward(self, weights, epsilon=0.7):
        """ Checks whether the chosen weights are balanced or not. Based on the result
        the agent could be punished or rewarded for its actions. """

        if len(np.where(weights == 0)[0]) > 0:
            return -10 # punish
        elif len(np.where(weights > epsilon)[0]) > 0:
            return -10 # punish
        elif len(np.where(weights == 1/self._number_of_coins)[0]) == self._number_of_coins:
            return 10
        else:
            return 0.5
    
    """def set_batch_size(self, batch_size):
        self._batch_size = batch_size"""
        
    def reset_weight_storage(self):
        # Balanced weights
        self._weight_storage = [self.initial_weights()]
        
    def initial_weights(self):
        """ Returns uniformely balanced weights were the capital is equally distributed 
        across the entire portfolio """
        #initial_weights = np.zeros(self._number_of_coins + 1); initial_weights[0] = 1
        initial_weights = np.ones(self._number_of_coins + 1) / (self._number_of_coins + 1)
        return initial_weights
    
    def overwrite_synthetic_data(self, recycle, overwrite, year):
        """ Regenerates the synthetic data again (for example if the crash parameters have been changed) and 
        overwrites the files in the synthetic_data folder of the specified year """
        
        if recycle is False and overwrite is True:
            print("Overwriting old synthetic crash files... ")
            for key, value in self.DataManager._portfolio.items():
                value.to_csv(f"cryptoportfolio/data/synthetic_data/{year}/{key}_{year}_synthetic.csv")
                
    def new_parameters(self, features, window_size):
        """ Receives a new list of features which are given to the DataManager in order to get the correct 
        price tensor. The observation space is also adapted. """
        
        assert self._features == ["close", "low", "high"], "Only works if no features were passed when initializing the agent."
        self._features = ["close", "low", "high"] + features
        self._feature_number = len(self._features)
        self._lookback_window_size = window_size
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._feature_number,
                                                                   self._number_of_coins, 
                                                                   self._lookback_window_size), 
                                            dtype=np.float32)
        
        # DataManager features are updated
        self.DataManager._features = self._features
        self.DataManager._feature_number = len(self._features)
        self.DataManager._lookback_window_size = window_size
        
    def reset_parameters(self):
        """ Receives a new list of features which are given to the DataManager in order to get the correct 
        price tensor. The observation space is also adapted. """
        
        if self._features != ["close", "low", "high"]:
            self._features = ["close", "low", "high"]
            self._feature_number = len(self._features)
            
            # DataManager features are reset
            self.DataManager._features = self._features
            self.DataManager._feature_number = len(self._features)