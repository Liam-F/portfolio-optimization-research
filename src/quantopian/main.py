import numpy as np
import pandas as pd
import preprocessing as pr
import features as ft
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

def filter_pairs(pairs):
    return pr.filter_on_nb_trades(
        pr.filter_on_rolling_sharpe(pairs, 0.8, 252)
    )


def create_inputs(pairs, features_list):
    nb_strategies = pairs.shape[1]
    X = np.zeros((nb_strategies, len(features_list)))
    for i, strategy in enumerate(pairs.columns):
        X[i] = pr.compute_features(pairs['2013':'2014'][strategy], features_list)
    return X


def create_outputs(pairs, feature_list):
    nb_strategies = pairs.shape[1]
    y = np.zeros((nb_strategies, 1))
    for i, strategy in enumerate(pairs.columns):
        y[i] = np.array([ft.sharpe_ratio(pairs['2015'])])
    return y


def main():
    pairs = pd.read_csv('./data/2017-08-03-filtered-in-sample-pairs.csv', parse_dates=True, index_col=0)
    pairs = pairs[pairs.columns[:100]]

    features_list = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                     ft.skewnewss, ft.kurtosis, ft.information_ratio,
                     ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                     ft.tail_ratio, ft.common_sense_ratio)

    X = create_inputs(pairs, features_list)
    X = scale(X, axis=1)
    y = create_outputs(pairs, features_list)
    y = scale(y, axis=1)

    forest = RandomForestRegressor()
    cross_val_score(forest, X, y, cv=4)

if __name__ == '__main__':
    main()