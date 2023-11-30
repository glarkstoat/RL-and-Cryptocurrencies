import numpy as np

def sharpe_ratio(rates_of_return):
        """ Formula (28) in (A deep learning framework for ...). Takes into account the risk of 
        a portfolio. The higher the better. """
        return np.mean(rates_of_return) / np.std(rates_of_return)
    
def max_drawdown(portfolio_values):
    """ Formula (29) in (A deep learning framework for ...). Returns the biggest lost form a 
    peak to a trough. The higher the bigger the maximum loss is i.e. the smaller the better. """

    drawdown_list = []
    p_t = 0
    for p_tau in portfolio_values:
        if p_tau > p_t:
            # No loss but gain, so p_t is the new peak and the loop continues to search for
            # the next trough
            p_t = p_tau
            drawdown_list.append(0.0) # means that there is no loss
        else:
            # Loss, so this is the trough and we need to calculate the drawdown. Continues 
            # until a new peak is found and p_t is overwritten again
            drawdown_list.append((p_t - p_tau) / p_t)
            
    return np.max(drawdown_list)