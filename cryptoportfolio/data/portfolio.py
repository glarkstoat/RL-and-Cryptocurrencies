import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sstudentt import SST

import seaborn as sns
sns.set_style('darkgrid')
sns.set(color_codes=True)
sns.set_context('talk', font_scale=0.5)

from cryptoportfolio.data.data_utils import origin, TI_calculation, crash_seeds, statistics


class Portfolio:
    """
    A portfolio containing all the OHCL-prices of the associated coins.
    If synthetic is set to True, it generates the synthetic portfolio according to the
    passed crash parameters file. 
    """
    
    def __init__(
        self, 
        year : str = "2021", 
        synthetic : bool = False, 
        recycle : bool = True, 
        split : str = "whole", 
        features : list = ["close", "low", "high"]
    ):
        
        self._parameters = []
        self._portfolio = {}
        assert year in ["2018", "2021", "2022", "Bull", "scenario4"], "Keyword <year> must be in ['2018', '2021', '2022', 'Bull', 'scenario4']."
        assert type(synthetic) is bool, "Keyword <synthetic> must be a boolean value."
        assert type(recycle) is bool, "Keyword <recycle> must be a boolean value."
        assert split in ["train", "validation", "whole"], "Keyword <split> must be in ['train', 'validation', 'whole']."
        if (year == "Bull" or year == "scenario4") and synthetic == True:
            raise ValueError("There is no synthetic crash for the selected year. Select synthetic=False.")
        self._synthetic = synthetic
        self._year = year
        self._recycle = recycle
        self._split = split
        self._features = features
        self._feature_number = len(features)
                
        year = f"cryptoportfolio/data/analysis/crash_parameters/{year}.csv"
        self.generate_portfolio(year)
       
    def generate_portfolio(self, year, plot=False):
        """ Takes a filename as an argument and generates a portfolio 
            following the parameters of the file. The portfolio is a dictionary
            where the arguments are the names of the coins (e.g. "BTC") and the values 
            are the OHCL prices of the coin. """

        self._parameters = pd.read_csv(year, header=0)
        self._number_of_coins = len(self._parameters)
        
        #status = "synthetic" if self._synthetic else "real"
        #print(f"Building {status} portfolio...")
        
        # Builds the portfolio
        self._coins = []
        for i in range(self._number_of_coins):
            if self._synthetic == True:
                
                # Generate synthetic OHCL prices
                row = self._parameters.iloc[i]
                coin = SyntheticCoin(row)
                
                # Add the coin names to the coins list
                self._coins.append(row['coin'])
                
                # To save on computation time, load the saved synthetic crash data from the repository (2018 instead 
                # of 20mins only takes 2 seconds). If synthetic data should be created again, just pass recycle as False.
                if self._recycle == True:  
                    self._portfolio[f"{row['coin']}"] = pd.read_csv(f"cryptoportfolio/data/synthetic_data/{self._year}/{row['coin']}_{self._year}_synthetic.csv", 
                                                                    header=0,
                                                                    usecols=self._features+["open"])
                    if self._split == "train":
                        self._portfolio[f"{row['coin']}"] = self._portfolio[f"{row['coin']}"][:int(len(self._portfolio[f"{row['coin']}"]) * 0.8)]
                    elif self._split == "validation":
                        self._portfolio[f"{row['coin']}"] = self._portfolio[f"{row['coin']}"][int(len(self._portfolio[f"{row['coin']}"]) * 0.8):].reset_index(drop=True)

                elif self._recycle == False:
                    self._portfolio[f"{row['coin']}"] = coin.generate_OHCL_prices(plot=plot)

            else:
                # Simply load the historic OHCL prices
                row = self._parameters.iloc[i]
                
                # Add the coin names to the coins list
                self._coins.append(row['coin'])
                
                self._portfolio[f"{row['coin']}"] = pd.read_csv(f"cryptoportfolio/data/raw/{row['coin']}_USDT_{origin[row['coin']]}_1h.csv", 
                                        header=0, 
                                        index_col=0, 
                                        parse_dates=True, 
                                        usecols=["time"]+self._features+["open"]
                                        ).iloc[row["start"]:row["end"]]
                if self._split == "train":
                    self._portfolio[f"{row['coin']}"] = self._portfolio[f"{row['coin']}"][:int(len(self._portfolio[f"{row['coin']}"]) * 0.8)]
                elif self._split == "validation":
                    self._portfolio[f"{row['coin']}"] = self._portfolio[f"{row['coin']}"][int(len(self._portfolio[f"{row['coin']}"]) * 0.8):]
            
    def get_global_tensor(self, features):
        
        crash_length = len(self._portfolio[list(self._portfolio.keys())[0]])
        tensor = np.zeros((self._feature_number, self._number_of_coins, crash_length)).astype(np.float32)
        # Writing the features for all coins to the tensor
        for i, feature in enumerate(features):
            for j, coin in enumerate(self._coins):
                # Write the row to the tensor
                tensor[i][j] = self._portfolio[coin][feature][:]

        return tensor

    def keys(self):
        return self._coins

    def plot(self, lw=0.5, alpha=.5):
        
        for key, value in self._portfolio.items():  
            #plt.figure(figsize=(7,4))   
            
            if self._synthetic == True:
                plt.title("Synthetic " + key + " Crash", fontweight="bold")
            elif self._synthetic == False:
                plt.title("Historic " + key + " Crash", fontweight="bold")

            plt.plot(value["high"], label="high", lw=lw, alpha=alpha, color="g")
            plt.plot(value["open"], label="open", lw=lw, alpha=alpha, color="r")
            plt.plot(value["close"], label="close", lw=lw, alpha=alpha, color="sandybrown")
            plt.plot(value["low"], label="low", lw=lw, alpha=alpha, color="b")
            plt.ylabel("USDT")
            #plt.xlabel("Trading Period")
            plt.legend()
            plt.tight_layout()
            #if key == "DOT":
            #    plt.savefig(f"dot_crash_{self._synthetic}.png", dpi=600)
            plt.show()


class SyntheticCoin:
    """ 
        Synthetic coin according to the passed parameters.
    
        coin : pd.Series
        A row for a particular crypto currency from the chosen crash_parameters___.csv file.
    """
    
    def __init__(self, coin):
        assert type(coin) is pd.core.series.Series, "Coin must be of type pd.core.series."
        self.coin = coin
        self.ohcl_prices = pd.DataFrame()
    
    def generate_OHCL_prices(self, statistics=False, plot=False):
        """ Generates OHCL prices of a simulated crash for a given crypto currency """

        # Loading the real crash data
        crash_real, crash_length = self.load_data()
        starting_price = crash_real["close"][0]
        
        # Generate closing prices
        closing_prices = self.generate_closing_prices(crash_real, crash_length, starting_price)
            
        # Calculating all necessary distributions
        dist_low, dist_high, dist_open = self.calculate_distributions(crash_real)
                
        # Calculating low-, high-, and opening prices from closing prices
        low_prices, high_prices, opening_prices = self.calculate_OHL_prices(closing_prices, 
                                                                       dist_low, dist_high, dist_open)
        
        # Collecting all OHCL prices into a single DataFrame
        data = {"close": closing_prices, "high": high_prices,
                "low": low_prices, "open": opening_prices}
        self.ohcl_prices = pd.concat(data, axis = 1)   
        
        # Calculating the technical indicators
        self.ohcl_prices = TI_calculation(self.ohcl_prices)
        
        if statistics == True:
            self.price_statistics(crash_real)
           
        if plot == True:
            self.plot_OHCL()
        
        return self.ohcl_prices

    def load_data(self):
        
        real_data = pd.read_csv(f'cryptoportfolio/data/raw/{self.coin["coin"]}_USDT_{origin[self.coin["coin"]]}_1h.csv')
        crash_real = real_data.iloc[self.coin["start"]:self.coin["end"]].reset_index(drop=True)
        crash_length = self.coin["end"] - self.coin["start"]
        
        return crash_real, crash_length

    def generate_closing_prices(self, crash_real, crash_length, starting_price):
        
        np.random.seed(crash_seeds[self.coin["crash_seed_index"]])
        print(self.coin["coin"]); print(crash_seeds[self.coin["crash_seed_index"]])

        returns_close_real = (crash_real["close"]).pct_change().dropna()
        
        # Generating the close prices from a skewed student-t distribution
        dist_close = SST(mu = returns_close_real.mean()/self.coin["close_mu"], 
                        sigma = returns_close_real.std()/self.coin["close_sigma"], 
                        nu = self.coin["close_nu"], tau = self.coin["close_tau"]) # nu > 1 results in skewness > 0 and vice versa
                                                                    # tau >! 2 to closer to 2 the more kurtosis
                                            
        # Samples randomly from the generated distribution to get the closing prices
        returns_close_syn = dist_close.r(crash_length) # +1 ?? (len of real and syn are sometimes not equal!)

        # Generate prices
        closing_prices = pd.Series(starting_price*(1+returns_close_syn).cumprod())
        
        return closing_prices

    def calculate_distributions(self, crash_real):
        
        # -------- Low -------- #
        np.random.seed(crash_seeds[self.coin["crash_seed_index"]])
        
        percentage_deviations_low_real = (crash_real["close"] - crash_real["low"]) / crash_real["low"]

        # Distribution of low - close values
        dist_low = SST(mu =  percentage_deviations_low_real.mean()/self.coin["low_mu"],         
                    sigma = percentage_deviations_low_real.std()/self.coin["low_sigma"],
                        nu = self.coin["low_nu"], tau = self.coin["low_tau"])
            
        # ------- High ------- #
        # Generating the high distribution
        np.random.seed(crash_seeds[self.coin["crash_seed_index"]])

        percentage_deviations_high_real = ((crash_real["high"] - crash_real["close"]) / crash_real["close"]) # for high prices

        # Distribution of high - close values
        dist_high = SST(mu =  percentage_deviations_high_real.mean()/self.coin["high_mu"], 
                        sigma = percentage_deviations_high_real.std()/self.coin["high_sigma"],
                        nu = self.coin["high_nu"], tau = self.coin["high_tau"])

        # ------- Open ------- #
        np.random.seed(crash_seeds[self.coin["crash_seed_index"]])

        percentage_deviations_open_real = ((crash_real["close"] - crash_real["open"]) / crash_real["open"]) # for high prices

        # Estimating the distribution of open - close values
        dist_open = SST(mu =  percentage_deviations_open_real.mean()/self.coin["open_mu"], 
                        sigma = percentage_deviations_open_real.std()/self.coin["open_sigma"],
                        nu = self.coin["open_nu"], tau = self.coin["open_tau"])
        
        return dist_low, dist_high, dist_open

    def calculate_OHL_prices(self, closing_prices, dist_low, dist_high, dist_open):
        """ Calculates opening-, high-, and low prices from the closing prices """
        
        low_prices, high_prices, opening_prices = [], [], []
        np.random.seed(crash_seeds[self.coin["crash_seed_index"]])
        #s = np.random.randint(0, 100000000)
        #np.random.seed(s)
        #print("random seed:", s)
        epsilon = self.coin["epsilon"]

        for i in tqdm(range(len(closing_prices))):
            
            if i == 0 or i == len(closing_prices)-1: 
                # First and last value since there is no respective closing value
                open_value = closing_prices[i] / (dist_open.r(1)[0] + 1)
                opening_prices.append(open_value)
            else:
                # Open value is the closing value from the previous hour
                open_value = closing_prices[i-1]
                opening_prices.append(open_value)
                
            # Randomly sample value from the distribution and calculate the low value
            low_value = closing_prices[i] / (self.sampling(dist_low, epsilon) + 1)
            
            # The low value must always be the min value for every respective hour
            while low_value > opening_prices[i]:
                if (closing_prices[i] - opening_prices[i]) / opening_prices[i] > epsilon: # not possible to reach the necessary deviation because epsilon cuts off distribution

                    low_value = opening_prices[i]
                else:
                    low_value = closing_prices[i] / (self.sampling(dist_low, epsilon) + 1) # potential bottleneck if epsilon is chosen very small 
                    # because probability of choosing necessary value becomes increasingly smaller  
            low_prices.append(low_value)
            
            # Randomly sample value from the distribution and calculate the high value
            high_value = closing_prices[i] * (self.sampling(dist_high, epsilon) + 1)
            
            # The high value must always be the max value for every respective hour
            while high_value < opening_prices[i]:
                if (opening_prices[i] - closing_prices[i]) / closing_prices[i] > epsilon: # not possible to reach the necessary deviation

                    high_value = opening_prices[i]
                else:
                    high_value = closing_prices[i] * (self.sampling(dist_high, epsilon) + 1) # potential bottleneck if epsilon is chosen very small 
            high_prices.append(high_value)  

        # Converting the prices to pd.Series
        opening_prices = pd.Series(opening_prices)
        high_prices = pd.Series(high_prices)
        low_prices = pd.Series(low_prices)
        
        return low_prices, high_prices, opening_prices

    def sampling(self, dist, epsilon):
        #s = np.random.randint(0, 100000000)
        #np.random.seed(s)
        #print("random seed:", s)
        sample = dist.r(1)[0]
        
        # Prevent deviations greater than epsilon
        while sample > epsilon:
            #return 0.001
            sample = dist.r(1)[0]
            
        # Since the sampling function is only used for low and high values, 
        # it is necessary to ensure that the sample is always positive to 
        # ensure that the low value is always the lowest and the high value
        # is always the highest at any given time
        if sample < 0:
            return 0
        else:
            return sample
        
    def price_statistics(self, crash_real):
        """ Statistics for the generated prices series """
        
        prices = {"close":self.ohcl_prices["close"],"open":self.ohcl_prices["open"],
                  "high":self.ohcl_prices["high"],"low":self.ohcl_prices["low"]}

        for price, series in prices.items():
            print("\nComparison of", price, ":")
            
            returns_real = (crash_real[price]).pct_change().dropna() # for single price series
            returns_syn = (series).pct_change().dropna() # for single price series

            statistics(returns_real, returns_syn)
            
            if price == "high":
                percentage_deviations_high_real = (crash_real["high"] - crash_real["close"]) / crash_real["close"] # for high prices
                percentage_deviations_high_syn = (self.ohcl_prices["high"] - self.ohcl_prices["close"]) / self.ohcl_prices["close"] # for high prices

                statistics(percentage_deviations_high_real, percentage_deviations_high_syn)
                
            elif price == "low":
                percentage_deviations_low_real = (crash_real["close"] - crash_real["low"]) / crash_real["low"] # for high prices
                percentage_deviations_low_syn = (self.ohcl_prices["close"] - self.ohcl_prices["low"]) / self.ohcl_prices["low"] # for high prices

                statistics(percentage_deviations_low_real, percentage_deviations_low_syn)

    def plot_OHCL(self, lw=.5, alpha=.8):
        
        plt.title("Synthetic Data")
        plt.plot(self.ohcl_prices["low"], label="low", lw=lw, alpha=alpha)
        plt.plot(self.ohcl_prices["open"], label="open", lw=lw, alpha=alpha)
        plt.plot(self.ohcl_prices["high"], label="high", lw=lw, alpha=alpha)
        plt.plot(self.ohcl_prices["close"], label="close", lw=lw, alpha=alpha)
        plt.legend()
        plt.show()