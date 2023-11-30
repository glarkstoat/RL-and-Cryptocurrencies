# Cryptocurrency Portfolio Management During A Market Crash Using Deep Reinforcement Learning
Master thesis for Computational Science (University of Vienna). 

This work was inspired by the Paper ["A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"](https://arxiv.org/abs/1706.10059) by Jiang et. al. 
Their source code can be found [here](https://github.com/ZhengyaoJiang/PGPortfolio). 

In my work I implemented two trading agents using PPO and DDPG respectively, which managed a portfolio of 13 cryptocurrecies during the 2022 cryptocurrency market crash, while seeking to maximize the return and also seeking to outperform the popular benchmarks and trading strategies in portfolio management.
The trading environment was built using [Gym OpenAI](https://www.gymlibrary.dev/), while the reinforcement learning models where built using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) and PyTorch.

## Data
The data was collected using the [CryptoCompare API](https://min-api.cryptocompare.com/), which provided the data for the most-traded cryptocurrencies from 2017-2022 with a temporal resolution of 60 minutes.

The cryptocurrency market experienced major crashes in 2018, 2021 and 2022. 

<p align="center">
  <img src="https://github.com/glarkstoat/RL-and-Cryptocurrencies/assets/74681570/d075cffc-45dd-45b9-b696-ab67af164abf" width="500" />
</p>

