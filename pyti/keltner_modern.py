from __future__ import absolute_import

import numpy as np
import talib

from pyti import catch_errors
from pyti.exponential_moving_average import (
    exponential_moving_average as ema
)
from pyti.simple_moving_average import (
    simple_moving_average as sma
)
from pyti.typical_price import typical_price


def band_width(high_data, low_data, period, exp=False):
    """
    Bandwidth.

    Formula:
    BW = SMA(H - L)
    """
    catch_errors.check_for_input_len_diff(high_data, low_data)
    diff = np.array(high_data) - np.array(low_data)

    if exp:
        bw = ema(diff, period)
    else:
        bw = sma(diff, period)

    return bw


def center_band(close_data, high_data, low_data, period, exp=False):
    """
    Center Band.

    Formula:
    CB = SMA(TP)
    """
    tp = typical_price(close_data, high_data, low_data)

    if exp:
        cb = ema(tp, period)
    else:
        cb = sma(tp, period)

    return cb


def upper_band(close_data, high_data, low_data, period, exp=True, **kwargs):
    """
    Upper Band.

    Formula:
    UB = CB + BW
    """
    cb = center_band(close_data, high_data, low_data, period, exp)
    bw = talib.ATR(np.array(high_data), np.array(low_data), np.array(close_data), period - 1) * kwargs.get("atrs", 2)
    ub = cb + bw
    return ub


def lower_band(close_data, high_data, low_data, period, exp=True, **kwargs):
    """
    Lower Band.

    Formula:
    LB = CB - BW
    """
    cb = center_band(close_data, high_data, low_data, period, exp)
    bw = talib.ATR(np.array(high_data), np.array(low_data), np.array(close_data), period - 1) * kwargs.get("atrs", 2Ø)
    lb = cb - bw
    return lb
