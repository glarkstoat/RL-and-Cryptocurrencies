from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
from ta.trend import SMAIndicator
from ta.momentum import StochasticOscillator
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import skew
from scipy.stats import kurtosis

api_key = 'a2c1563f44eba234148141fd1da424274e3ec468a48c28408c88a89bb2bc19c9'

origin = {"BTC":"Poloniex", "ETH":"Poloniex", "LTC":"Poloniex", "NEO":"Binance", "BNB":"Binance", "XRP":"Poloniex",
         "LINK":"Binance", "EOS":"Gateio", "TRX":"Binance", "ETC":"Poloniex", "XLM":"Poloniex", "SOL":"Binance",
         "ADA":"Gateio", "QTUM":"Gateio", "DASH":"Poloniex", "XMR":"Poloniex", "DOT":"Binance", "BCH":"Gateio", "BSV":"CoinEx",
         "LEO":"Bitfinex", "LUNA":"Binance", "XEM":"Gateio", "DOGE":"Poloniex", "MIOTA":"Gateio", "CRO":"DigiFinex", "SHIB":"Poloniex",
         "AVAX":"Binance", "MATIC":"Binance"}

# Random seeds that produce realistic looking crashes
# ONLY APPEND NEW SEEDS TO THE END. DO NOT CHANGE THE ORDER OF THE SEEDS. THIS WOULD SCREW UP 
# THE INDICES IN THE CRASH_PARAMETERS.CSV FILE!
crash_seeds = [1,2,9,10,17,18,23,28,45,54,59,61,62,64,66,68,76,81,84,90,92,94,95,100,104,118,
               129,131,135,142,147,151,152,157,175,179,187,200,203,227,255,277,311,341,348,363,
               385,433,447,462,468,483,487,491,511,513,518,528,546,547,549,580,582,584,591,594,599,613,615,
               60,65,79,112,144,145,195,230,289,290,297,312,379,383,464,41,124,170,12,116,168,192,715,724,
               779,739,681]

def TI_calculation(data):
    """ Calculation of technical indicators """
    
    # Moving Average Convergence Divergence (MACD)  
    MACD = MACDIndicator(data["close"])
    data["mcd"] = MACD.macd()
    data["mcd_signal"] = MACD.macd_signal()
    
    # Relative Strength Index (RSI)
    RSI = RSIIndicator(data["close"])
    data["RSI"] = RSI.rsi()

    # Simple Moving Average (SMA)
    SMA_50 = SMAIndicator(data["close"], window=50)
    SMA_200 = SMAIndicator(data["close"], window=200)
    data["SMA_50"] = SMA_50.sma_indicator()
    data["SMA_200"] = SMA_200.sma_indicator()    
    
    # Stochastic Oszillator
    STOCH = StochasticOscillator(data["high"], data["low"], data["close"])
    data["stoch_oszillator"] = STOCH.stoch()
    data["stoch_oszillator_signal"] = STOCH.stoch_signal()
    
    # Bollinger Bands
    Bollinger = BollingerBands(data["close"])
    data["Bollinger_middle"] = Bollinger.bollinger_mavg()
    data["Bollinger_low"] = Bollinger.bollinger_lband()
    data["Bollinger_high"] = Bollinger.bollinger_hband() 
    
    # Replace every NaN value with 0
    data = data.fillna(0)
    
    return data  

def statistics(returns_real, returns_syn, qqplot=True, cdf=True):
    """ Basic statistical analysis of given returns """
    
    #plt.title("Histogram")
    plt.hist(returns_real, bins='auto', label="real", alpha=0.2)#, color="black")
    plt.hist(returns_syn, bins='auto', label="syn", alpha=0.2)#, color="green")
    plt.xlim([-0.1,0.1])
    plt.legend()
    plt.show()

    if qqplot == True:
        
        t = np.linspace(0.01,0.99,1000)
        q1 = np.quantile(returns_real,t)
        q2 = np.quantile(returns_syn,t)
        plt.title("Q-Q Plot")
        plt.plot(q1, q2)
        plt.plot([min(q1),max(q1)],[min(q2),max(q2)])
        plt.xlim((min(q1),max(q1)))
        plt.ylim((min(q2),max(q2)))
        plt.xlabel("Returns")
        plt.show()
        
    if cdf == True:

        plt.title("CDF Plot")
        plt.plot(np.sort(returns_real), 1. * np.arange(len(returns_real)) / (len(returns_real) - 1), label="Real")
        plt.plot(np.sort(returns_syn), 1. * np.arange(len(returns_syn)) / (len(returns_syn) - 1), label="Syn")
        plt.legend()
        plt.show()
 
    print("For approximated distribution: ")
    print("Mean: ", returns_real.mean(), " (real) | ", returns_syn.mean(), " (syn)")
    print("Std: ", returns_real.std(), " (real) | ", returns_syn.std(), " (syn)")
    print("Skewness: ", skew(returns_real), " (real) | ", skew(returns_syn), " (syn)")
    print("Kurtosis: ", kurtosis(returns_real), " (real) | ", kurtosis(returns_syn), " (syn)")