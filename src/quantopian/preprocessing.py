import numpy as np
import features as ft


def filter_on_rolling_sharpe(returns, threshold, window=252):
    one_year_sharpe = returns.rolling(window).apply(ft.sharpe_ratio)
    return returns[returns.columns[(one_year_sharpe > threshold).any()]]


def filter_on_nb_trades(returns, percent=0.1):
    number_days = returns.shape[0]
    threshold = int(number_days * percent)
    return returns[returns.columns[returns[returns == 0].count() > threshold]]


def compute_features(returns, feature_functions):
    result = []
    for feature in feature_functions:
        result.append(feature(returns[returns != 0]))
    return np.array(result)
