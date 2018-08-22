from __future__ import absolute_import
import numpy as np
from pyti import catch_errors
from pyti.simple_moving_average import (
    simple_moving_average as sma
)
from pyti.exponential_moving_average import (
    exponential_moving_average as ema
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


def upper_band(close_data, high_data, low_data, period, exp=False, **kwargs):
    """
    Upper Band.

    Formula:
    UB = CB + BW
    """
    cb = center_band(close_data, high_data, low_data, period, exp)
    bw = band_width(high_data, low_data, period, exp)
    ub = cb + bw
    return ub


def lower_band(close_data, high_data, low_data, period, exp=False, **kwargs):
    """
    Lower Band.

    Formula:
    LB = CB - BW
    """
    cb = center_band(close_data, high_data, low_data, period, exp)
    bw = band_width(high_data, low_data, period, exp)
    lb = cb - bw
    return lb
