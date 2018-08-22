from __future__ import absolute_import
import numpy as np
from pyti import catch_errors
from pyti.function_helper import fill_for_noncomputable_vals
from six.moves import range

# import pandas as pd


# legacy Pandas (too slow for backtesting)
# def pandas_smoothed_moving_average(data, period):
#     """
#     Smoothed Moving Average.
#
#     Formula:
#     smma = avg(data(n)) - avg(data(n)/n) + data(t)/n
#     """
#     catch_errors.check_for_period_error(data, period)
#     series = pd.Series(data)
#     return series.ewm(alpha = 1.0/period).mean().values.flatten()


# faster alternative implementations
###########################################
def smoothed_moving_average(data, window):
    data = np.array(data)
    alpha = 2 / (window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def alt2_smoothed_moving_average(data, period):
    """
      Returns the exponentially weighted moving average of x.

      Parameters:
      -----------
      data : array-like
      period : float {0 <= period <= 1}

      Returns:
      --------
      ewma : numpy array
          the exponentially weighted moving average
    """
    # coerce x to an array
    data = np.array(data)
    n = data.size
    # create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (1 - period)
    p = np.vstack([np.arange(i, i - n, -1) for i in range(n)])
    # create the weight matrix
    w = np.tril(w0 ** p, 0)
    # calculate the ewma
    return np.dot(w, data[::np.newaxis]) / w.sum(axis=1)


def alt3_smoothed_moving_average(data, window):
    data = np.array(data)
    alpha = 2 / (window + 1.0)
    scale = 1/(1-alpha)
    n = data.shape[0]
    scale_arr = (1-alpha)**(-1*np.arange(n))
    weights = (1-alpha)**np.arange(n)
    pw0 = (1-alpha)**(n-1)
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = cumsums*scale_arr[::-1] / weights.cumsum()

    return out
