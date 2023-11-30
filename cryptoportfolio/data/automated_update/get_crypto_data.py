# Gets the newest version of the data set (max. last 2000 hours), manipulates it and saves it to a csv file.
# %%
import requests
import pandas as pd
import os, sys
from win10toast import ToastNotifier
from datetime import datetime

sys.path.insert(0, 'C:\\Users\\chris\\Documents\\GitHub\\RL-and-Cryptocurrencies\\cryptoportfolio\\data')
from data_utils import api_key
from data_utils import TI_calculation
from data_utils import origin

# Windows notification
toast = ToastNotifier()
toast.show_toast("Crypto Data Update", "The update has been started", duration=30)
os.chdir(r"C:\Users\chris\Documents\GitHub\RL-and-Cryptocurrencies\cryptoportfolio\data\raw")

for symbol in origin.keys():
    ''' Automatic update of all currencies, fetching missing data between the present date and the 
    last recorded date in the respective dataset '''

    conversion = "USDT"
    coin = pd.read_csv(f"{symbol}_{conversion}_{origin[symbol]}_1h.csv", index_col=0, parse_dates=True).iloc[-1]

    present = datetime.now()
    present = datetime(present.year, present.month, present.day, present.hour)
    earliest_date = coin.name
    
    # Difference in hours
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

            data.to_csv(f"{symbol}_USDT_{origin[symbol]}_1h.csv", mode='a', index=False, header=False)
            
            # Calculate TIs for the whole dataset from scratch (easier than just for the new rows ...)
            coin = pd.read_csv(f"{symbol}_USDT_{origin[symbol]}_1h.csv", index_col=0, parse_dates=True)
            TI_calculation(coin).to_csv(f"{symbol}_USDT_{origin[symbol]}_1h.csv")

        except:
            toast.show_toast("ERROR: Crypto Data Update", f"Couldn't update data for {symbol}. \nError message: {data['Message']}", duration=300)
    else:
        toast.show_toast("ERROR: Crypto Data Update", f"The update exceeds limit of 2000 data points for {symbol}", duration=300)

toast.show_toast("Crypto Data Update Completed", "The update has been completed", duration=30)
# %%

