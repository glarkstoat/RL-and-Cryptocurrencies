import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from cryptoportfolio.data.portfolio import Portfolio


class DataManager:
    """ Handles everything concerning the portfolio data """
    
    def __init__(
        self, 
        lookback_window_size : int = 50, 
        features : list = ["close", "low", "high"]
    ):
        
        self._lookback_window_size = lookback_window_size
        self._features = features
        self._feature_number = len(self._features)
        self._portfolio : np.array = None
        self._global_tensor : np.array = None

    def generate_portfolio(self, year="2021", synthetic=False, recycle=True, split="whole"):
        """ Takes a filename as an argument and generates a portfolio 
            following the parameters of the file. The portfolio is a dictionary
            where the arguments are the names of the coins (e.g. "BTC") and the values 
            are the OHCL prices of the coin. """
        
        self._portfolio = Portfolio(year=year, synthetic=synthetic, recycle=recycle, split=split, features=self._features)
        self._global_tensor = self._portfolio.get_global_tensor(self._features)
        self._number_of_coins = self._global_tensor.shape[1] # number of non-cash assets
        self._crash_length = self._global_tensor.shape[2]

        return self._crash_length

    def price_tensor(self, timestep):
        """ For a given time step, returns the price tensor containing the normalized closing, lowest and highest values 
        for the predefined lookback window e.g. for the last 50 data points from the given time step. """
        
        assert self._portfolio is not None, "Call the generate_portfolio function first."
        assert timestep >= self._lookback_window_size - 1 and timestep <= self._crash_length - 1, (
            f"Timestep is only defined within the interval [{self._lookback_window_size - 1}, "
            f"{self._crash_length - 1}]. Current timestep: {timestep}."
        )
        
        start = timestep - self._lookback_window_size + 1
        end = timestep + 1
    
        tensor = get_obs_tensor(self._global_tensor, timestep, start, end,
                                self._lookback_window_size,
                                self._number_of_coins,
                                self._feature_number)
        return tensor

    
    def relative_price_vector(self, timestep):
        # Relative price vector for a given timestep
        assert self._portfolio is not None, "Call the generate_portfolio function first."
        
        if timestep == 0:
            # Necessary because otherwise the closing prices are normalized by global_tensor[0][.][-1]!
            y = np.ones((self._number_of_coins,))
        else:
            y = np.array([self._global_tensor[0][i][timestep] / self._global_tensor[0][i][timestep-1] for i in range(self._number_of_coins)])
        
        # Inserts the change of the cash asset, which is always 1, at the start of the array
        y = np.insert(y, 0, 1)

        return y
                
    def plot(self, lw=.5, alpha=.7):
        
        assert self._portfolio is not None, "Call the generate_portfolio function first."
        
        for i, coin in enumerate(self._portfolio._coins):
            plt.title(coin)
            plt.plot(self._global_tensor[1][i], label="high", lw=lw, alpha=alpha)
            
            open_prices = np.insert(self._global_tensor[0][i], 0, self._global_tensor[0][i][0])
            plt.plot(open_prices, label="open", lw=lw, alpha=alpha)

            plt.plot(self._global_tensor[0][i], label="close", lw=lw, alpha=alpha)
            plt.plot(self._global_tensor[2][i], label="low", lw=lw, alpha=alpha)
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        plt.title("Relative Closing Price Changes")
        for i, coin in enumerate(self._portfolio._coins):
            plt.plot(self._global_tensor[0][i]/self._global_tensor[0][i][0], label=coin, lw=lw, alpha=alpha)
            plt.legend()
        plt.tight_layout()
        plt.show()

@jit(nopython=True, fastmath=True)
def get_obs_tensor(global_tensor, timestep, start, end, 
                   window_size, number_of_coins, feature_number):
    """ Returns the normalized price tensor for a given timestep. 
        Global function, so that jit can do its work. """

    obs_tensor = np.zeros((feature_number, 
                            number_of_coins,
                            window_size)).astype(np.float32)

    """
    # Normalization
    for i in range(feature_number):
        for j in range(number_of_coins):
            obs_tensor[i, j,:] = global_tensor[i,j,start:end] / global_tensor[i,j,timestep]
    """

    # Normalization
    for i in range(feature_number):
        for j in range(number_of_coins):
            if i > 2:
                if global_tensor[i,j,timestep] != 0:
                    # Normalize the technical indicators
                    obs_tensor[i,j,:] = global_tensor[i,j,start:end] / global_tensor[i,j,timestep]
            else:
                # Normalize closing, high and low prices by the closing prices at given timestep
                obs_tensor[i,j,:] = global_tensor[i,j,start:end] / global_tensor[0,j,timestep]

    return obs_tensor
