#%%

import bitfinex
import datetime
import time
import pandas as pd
from datetime import date
from datetime import timedelta
import requests
from datetime import datetime
import os 
os.chdir(r"C:\Users\chris\Documents\GitHub\RL-and-Cryptocurrencies\Data")

#%% Bitfinex API
# Define query parameters
pair = 'MIOTAUST' # Currency pair of interest
TIMEFRAME = '1h'#,'4h','1h','15m','1m'
TIMEFRAME_S = 3600 # seconds in TIMEFRAME

# Get yesterdays date
yesterday = date.today() - timedelta(days = 1)

# Define the start
t_start = datetime(2017, 
                            6, 
                            1, 
                            1, 0)
t_start = time.mktime(t_start.timetuple()) * 1000

# Define the end
t_stop = datetime(yesterday.year, 
                           yesterday.month, 
                           yesterday.day, 
                           23, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000

def fetch_data(start, stop, symbol, interval, TIMEFRAME_S):
    limit = 1000    # We want the maximum of 1000 data points
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    hour = TIMEFRAME_S * 1000
    step = hour * limit
    data = []

    total_steps = (stop-start)/hour
    while total_steps > 0:
        if total_steps < limit: # recalculating ending steps
            step = total_steps * hour

        end = start + step
        data += api_v2.candles(symbol=symbol, interval=interval, limit=limit, start=start, end=end)
        #print(pd.to_datetime(start, unit='ms'), pd.to_datetime(end, unit='ms'), "steps left:", total_steps)
        start = start + step
        total_steps -= limit
        time.sleep(1.5)
    return data

result = fetch_data(t_start, t_stop, pair, TIMEFRAME, TIMEFRAME_S)
names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
df = pd.DataFrame(result, columns=names)
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df.to_csv(f"{pair}_{TIMEFRAME}.csv")

# %% Poloniex API

# import this package
from poloniex import Poloniex

# make an instance of poloniex.Poloniex
polo = Poloniex()

start = datetime(2017, 
                            1, 
                            1, 
                            0, 0)

#stop = start + timedelta(days=577)
start = datetime.timestamp(start) #time.mktime(start.timetuple())

stop = datetime(2018, 
                            2, 
                            16, 
                            0, 0)

stop = datetime.timestamp(stop) #time.mktime(stop.timetuple())

res = pd.DataFrame(polo.returnChartData(currencyPair='USDT_QTUM', period=1800, 
                           start=start,
                           end=stop))
res = res.astype({'close': float, 'open': float, 
                  'high': float, 'low': float,
                  'volume': float, 'date': int})
res["date"] = [datetime.fromtimestamp(date) for date in res["date"]]
res = res.astype({'date': str})

# %% Coin API

from coinapi_rest_v1.restapi import CoinAPIv1
import sys

test_key = "213B9265-D9D3-4905-8E4A-6B4ED5843A68"

api = CoinAPIv1(test_key)
exchanges = api.metadata_list_exchanges()
"""
print('Exchanges')
for exchange in exchanges:
    print('Exchange ID: %s' % exchange['exchange_id'])
    print('Exchange website: %s' % exchange['website'])
    print('Exchange name: %s' % exchange['name'])


assets = api.metadata_list_assets()
print('Assets')
for asset in assets:
    print('Asset ID: %s' % asset['asset_id'])
    try:
        print('Asset name: %s' % asset['name'])
    except KeyError:
        print('Can not find name')
    print('Asset type (crypto?): %s' % asset['type_is_crypto'])
"""
#%%

symbols = api.metadata_list_symbols()
print('Symbols')
for symbol in symbols:
    print('Symbol ID: %s' % symbol['symbol_id'])
    print('Exchange ID: %s' % symbol['exchange_id'])
    print('Symbol type: %s' % symbol['symbol_type'])
    try:
        print('Asset ID base: %s' % symbol['asset_id_base'])
    except KeyError:
        print('Can not find Asset ID base')
    try:
        print('Asset ID quote: %s' % symbol['asset_id_quote'])
    except KeyError:
        print('Can not find Asset ID quote')

    if (symbol['symbol_type'] == 'FUTURES'):
        print('Future delivery time: %s' % symbol['future_delivery_time'])

    if (symbol['symbol_type'] == 'OPTION'):
        print('Option type is call: %s' % symbol['option_type_is_call'])
        print('Option strike price: %s' % symbol['option_strike_price'])
        print('Option contract unit: %s' % symbol['option_contract_unit'])
        print('Option exercise style: %s' % symbol['option_exercise_style'])
        print('Option expiration time: %s' % symbol['option_expiration_time'])

periods = api.ohlcv_list_all_periods()

for period in periods:
    print('ID: %s' % period['period_id'])
    print('Seconds: %s' % period['length_seconds'])
    print('Months: %s' % period['length_months'])
    print('Unit count: %s' % period['unit_count'])
    print('Unit name: %s' % period['unit_name'])
    print('Display name: %s' % period['display_name'])

ohlcv_latest = api.ohlcv_latest_data('BITSTAMP_SPOT_BTC_USD', {'period_id': '1MIN'})

for period in ohlcv_latest:
    print('Period start: %s' % period['time_period_start'])
    print('Period end: %s' % period['time_period_end'])
    print('Time open: %s' % period['time_open'])
    print('Time close: %s' % period['time_close'])
    print('Price open: %s' % period['price_open'])
    print('Price close: %s' % period['price_close'])
    print('Price low: %s' % period['price_low'])
    print('Price high: %s' % period['price_high'])
    print('Volume traded: %s' % period['volume_traded'])
    print('Trades count: %s' % period['trades_count'])

# %%
ohlcv_historical = []
start = datetime(2017, 
                            10, 
                            1, 0, 0 
                            )
for i in range(1):
    ohlcv_historical += api.ohlcv_historical_data('POLONIEX_SPOT_ADA_USDT', {'period_id': '1HRS', 'time_start': start.isoformat()})

    start = start + timedelta(minutes=60*100)

# %%
start = date(2017, 1, 1)

ohlcv_historical = api.ohlcv_historical_data('POLONIEX_SPOT_ETH_USDT', {'period_id': '1HRS', 'time_start': start.isoformat()})
#%%
# CoinBase Pro API
import pandas as pd
import requests
import json

def fetch_daily_data(symbol, start, end):
    pair_split = symbol.split('/')  # symbol must be in format XXX/XXX ie. BTC/EUR
    symbol = pair_split[0] + '-' + pair_split[1]
    url = f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity=3600&start={start.isoformat()}&end={end.isoformat()}'
    response = requests.get(url)
    if response.status_code == 200:  # check to make sure the response from server is good
        data = pd.DataFrame(json.loads(response.text), columns=['unix', 'low', 'high', 'open', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['unix'], unit='s')  # convert to a readable date
        data['vol_fiat'] = data['volume'] * data['close']      # multiply the BTC volume by closing price to approximate fiat volume

        # if we failed to get any data, print an error...otherwise write the file
        if data is None:
            print("Did not return any data from Coinbase for this symbol")
        else:
            return data
    else:
        print("Did not receieve OK response from Coinbase API")
        
pair = "XRP/USDT"
start = datetime(2022, 1, 1, 0, 0)
end = start + timedelta(hours=290)
data = fetch_daily_data(pair, start, end)
#%%

"""
# Download the historical data for all pairs
for pair in pairs:
        
    # Downloads the latest version of the data set
    csv_url = f"https://www.cryptodatadownload.com/cdd/Binance_{pair}_1h.csv"
    req = requests.get(csv_url)
    url_content = req.content

    # Converts data to pandas dataframe and removes unnecessary column
    df = pd.read_csv(io.StringIO(url_content.decode('utf-8')), header=1).drop(columns=["unix"])

    df.to_csv(f'Binance_{pair}_1h.csv', index=False)
"""