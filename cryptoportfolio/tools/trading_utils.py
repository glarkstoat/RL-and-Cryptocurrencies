import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

sns.set_style('darkgrid')
sns.set(color_codes=True)
sns.set_context('talk', font_scale=0.65)

@jit(nopython=True, fastmath=True)
def calculate_reward(relative_price_vector, weights, transaction_factor):
    """ Calculates the reward for a given timestep """
    reward = np.log(relative_price_vector.astype(np.float32) @ weights.astype(np.float32) * transaction_factor)
    return reward
    
def transaction_factor(relative_price_vector, weights, commission_rate):
    """ weights are a list of lists of the two last weights """
    w_prime = (relative_price_vector.astype(np.float32) * weights[-2].astype(np.float32)) / (relative_price_vector.astype(np.float32) @ weights[-2].astype(np.float32))
    diff = torch.as_tensor(w_prime - weights[-1])
    sum_term = torch.abs(diff)[1:].sum() # sum of all non-cash elements (0th element is cash currency)
    mu = 1 - commission_rate * sum_term

    """ # Iterative approach
    mu = 1-3*c+c**2
    def recurse(mu0):
        factor1 = 1/(1 - c*w_t1[:, 0])
        if isinstance(mu0, float):
            mu0 = mu0
        else:
            mu0 = mu0[:, None]
        factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
            tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
        return factor1*factor2
    for i in range(20):
        mu = recurse(mu)
    return mu
    """        
    
    return mu.item(), sum_term.item() # just the float value

@jit(nopython=True, fastmath=True)
def portfolio_value(portfolio_value_old, transaction_factor, relative_price_vector, weights):
    pv = portfolio_value_old * transaction_factor * (relative_price_vector.astype(np.float32) @ weights.astype(np.float32))
    return pv
        
def plot_portfolio_values(portfolio_values):
    
    plt.figure(figsize=(12,7))
    plt.title("Portfolio Values")
    plt.xlabel("Steps")
    plt.ylim([-0.1,np.max(portfolio_values)+0.2])
    plt.plot(portfolio_values)
    plt.plot(np.linspace(0, len(portfolio_values)-1, len(portfolio_values)), 
                [1]*len(portfolio_values), color="black", lw=0.7, ls="--", alpha=0.5,
                label="Inital Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def normalize_weights(weights):
    
    if sum(weights) != 1.0:
        weights /= sum(weights)
        return weights
    else:
        return weights
    
def plot_weight_changes(weight_storage, number_of_coins, coins):
    
    plt.figure(figsize=(9,12))
    
    #r = int(number_of_coins/4)
    for i in range(1,number_of_coins+1):
        ax = plt.subplot(4,3, i)
        """if number_of_coins % 4 > 0:
            ax = plt.subplot(r+1,4,i)
        else:
            ax = plt.subplot(r,4,i)"""
        ax.set_title("Weights for "+str(list(coins)[i-1]), fontsize=13, fontweight="bold")
        ax.set_xlabel("Steps", fontweight="bold")
        ax.plot([x[i] for x in weight_storage], lw=3, alpha=0.8)
        #ax.set_ylim([-0.1, 1.1])#1.1
        ax.set_ylim([0, 0.1])
            #[min([x[i] for x in weight_storage])*0.4, max([x[i] for x in weight_storage])*1.1])
            
    """
    r = int(number_of_coins/4)
    for i in range(1,number_of_coins+1):
        if number_of_coins % 4 > 0:
            ax = plt.subplot(r+1,4,i)
        else:
            ax = plt.subplot(r,4,i)
        ax.set_title("Weights for "+str(list(coins)[i-1]), fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps")
        ax.plot([x[i] for x in weight_storage], lw=3, alpha=0.8)
        ax.set_ylim([-0.1, 1.1])
            #[min([x[i] for x in weight_storage])*0.4, max([x[i] for x in weight_storage])*1.1])
    """

    #plt.savefig(f"weights.png", dpi=600)
    plt.tight_layout()
    plt.show()
    
    # Bias
    plt.figure(figsize=(3,3))
    plt.title("Weights for Bias", fontsize=13, fontweight="bold")
    
    plt.plot([x[0] for x in weight_storage], lw=3, alpha=0.8)
    #plt.ylim([-0.1, 1.1])
    plt.ylim([0, 0.1])
    plt.xlabel("Steps", fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    #plt.figure(figsize=(5,3))
    #plt.title("Portfolio Weight Changes - Misc.")
    
    """plt.plot([x[1] for x in weight_storage][100:110], lw=3, alpha=0.8)
    #plt.ylim([-0.1, 1.1])
    #plt.xlim([300, 400])
    plt.xlabel("Steps")
    plt.tight_layout()
    plt.show()
    print("Std:", np.std([x[1] for x in weight_storage]))"""

def plot_rewards(rewards):
    
    plt.figure(figsize=(12,7))
    plt.title("Immediate Rewards")
    plt.xlabel("Steps")
    plt.plot(rewards, alpha=0.7, lw=0.4)
    plt.tight_layout()
    plt.show()      
    
def plot_transaction_costs(transaction_costs):
    plt.figure(figsize=(12,7))
    plt.title("Transaction Cost")
    plt.xlabel("Steps")
    plt.plot(transaction_costs, alpha=1, lw=2)
    plt.tight_layout()
    plt.show()  