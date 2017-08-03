import numpy as np
import pandas as pd
import preprocessing as pr
import features as ft


def filter_pairs(pairs):
    return pr.filter_on_nb_trades(
        pr.filter_on_rolling_sharpe(pairs, 0.8, 252)
    )


def main():
    pairs = pd.read_csv('./data/2017-08-03-filtered-in-sample-pairs.csv', parse_dates=True, index_col=0)

    features_list = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                     ft.skewnewss, ft.kurtosis, ft.information_ratio,
                     ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                     ft.tail_ratio, ft.common_sense_ratio)

    nb_strategies = pairs.shape[1]

    X = np.zeros((nb_strategies, len(features_list)))
    for i, strategy in enumerate(pairs.columns):
        X[i] = pr.compute_features(pairs['2013':'2014'][strategy], features_list)

    Y = np.zeros((nb_strategies, 1))
    for i, strategy in enumerate(pairs.columns):
        Y[i] = np.array([ft.sharpe_ratio(pairs['2015'])])




if __name__ == '__main__':
    main()