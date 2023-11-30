@echo off

python C:\Users\chris\Documents\GitHub\RL-and-Cryptocurrencies\cryptoportfolio\data\automated_update\get_crypto_data.py
cd C:\Users\chris\Documents\GitHub\RL-and-Cryptocurrencies\cryptoportfolio\data\raw
git add -A
git commit -m "data update"
git push

@pause