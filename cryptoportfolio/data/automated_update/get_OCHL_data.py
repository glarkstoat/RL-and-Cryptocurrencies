"""

DANGEROUS FILE!!!! DO NOT EXECUTE UNLESS ABSOLUTELY NECESSARY!!!!

"""

# %%
import bitfinex
import datetime
import time
import pandas as pd
from datetime import date
from datetime import timedelta
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, 'C:\\Users\\chris\\Documents\\GitHub\\RL-and-Cryptocurrencies\\cryptoportfolio\\data')
from utils import api_key
from utils import TI_calculation
from utils import origin
  
#%% Get entire data sets from CryptoCompare
os.chdir(r"C:\Users\chris\Documents\GitHub\RL-and-Cryptocurrencies\cryptoportfolio\data\raw")

for symbol in ["MATIC"]:#origin.keys():
    #symbol = 'DOT'
    conversion = 'USDT'
    exchange = origin[symbol]
    #exchange = "Binance"
    present = datetime.now()
    time_stamp = datetime(present.year, present.month, present.day, present.hour)
    earliest_date = datetime.timestamp(datetime(2017, 1, 1, 0))

    try:
        data_all = pd.DataFrame()
        for i in range(50):
            if datetime.timestamp(time_stamp) >= earliest_date:
                
                url = f'https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={conversion}&limit=2000&e={exchange}&toTs={datetime.timestamp(time_stamp)}&api_key={api_key}'
                r = requests.get(url)
                data = r.json()
                data = pd.DataFrame(data["Data"]["Data"])
                data["time"] = [datetime.fromtimestamp(dt) for dt in data["time"]]
                data_all = pd.concat([data, data_all], ignore_index=True)
            
                time_stamp = time_stamp - timedelta(hours=2002)

        data_all.drop(["conversionType", "conversionSymbol"], axis=1, inplace=True)
        data_all = data_all[(data_all.close > 0) & (data_all.high > 0) & (data_all.low > 0) & (data_all.open > 0)]

        # Calculation of technical indicators
        data_all = TI_calculation(data_all)
        
        # Writing data to csv-file
        data_all.to_csv(f'{symbol}_{conversion}_{exchange}_1h.csv', index=False)
    except:
        print(f"ERROR: with {symbol}")
#%% Get the missing data from the present to the last date in datasets of coins
os.chdir(r"C:\Users\chris\Documents\GitHub\RL-and-Cryptocurrencies\cryptoportfolio\data\raw")

for symbol in origin.keys():

    coin = pd.read_csv(f"{symbol}_USDT_{origin[symbol]}_1h.csv", index_col=0, parse_dates=True).iloc[-1]
    conversion = "USDT"

    present = datetime.now()
    present = datetime(present.year, present.month, present.day, present.hour)
    earliest_date = coin.name
    difference = (present - earliest_date).total_seconds() / 3600

    if difference <= 0:
        print(f"{symbol} is already updated.")
    elif difference < 2000:
        url = f'https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={conversion}&limit={1 if difference == 1 else int(difference - 1)}&e={origin[symbol]}&toTs={datetime.timestamp(present)}&api_key={api_key}'
        r = requests.get(url)
        data = r.json()
        try:
            data = pd.DataFrame(data["Data"]["Data"])
            data["time"] = [datetime.fromtimestamp(dt) for dt in data["time"]]

            data.drop(["conversionType", "conversionSymbol"], axis=1, inplace=True)
            
            # Weird behavior of API where limit=1 and limit=2 returns the last two data points
            if difference == 1:
                data = data[data["time"] == present]
                
            # Calculation of technical indicators
            data = TI_calculation(data)
            
            data.to_csv(f"{symbol}_USDT_{origin[symbol]}_1h.csv", mode='a', index=False, header=False)
        except:
            print(f"Couldn't update data for {symbol}. \nError message: {data['Message']}")
    else:
        print(f"Update exceeds limit of 2000 data points for {symbol}")
