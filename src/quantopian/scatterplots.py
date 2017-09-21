import pandas as pd

import quantopian.features as ft
import seaborn as sns
import matplotlib.pyplot as plt

from quantopian.main import compute_features, compute_labels


def __main__():
    production_strategies_pnls = pd.read_csv('./data/production-strategies.csv', parse_dates=True, index_col=0)
    pnls = production_strategies_pnls

    pnls = pd.read_csv('./data/all-pairs.csv', parse_dates=True, index_col=0)

    features = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                ft.skewness, ft.kurtosis, ft.information_ratio,
                ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                ft.tail_ratio, ft.common_sense_ratio,
                ft.sharpe_ratio_last_30_days, ft.sharpe_ratio_last_90_days,
                ft.sharpe_ratio_last_150_days)

    X = compute_features(pnls['2014':'2015'], features, drop_nans=True, scale=True)
    y = compute_labels(pnls['2016':'2016'], return_series=True).rename('OOS Sharpe')

    Xy = X.merge(y.to_frame(), left_index=True, right_index=True, how='left')
    Xy = Xy.query('annvol != 0 and annret != 0')
    X = Xy.drop('OOS Sharpe', axis=1)
    y = Xy['OOS Sharpe']

    sns.set_context('talk')
    for col in X:
        print('Plotting %s' % col)
        sns.jointplot(X[col], y, kind='reg')
        plt.savefig('./plots/scatterplots_v2_12kpairs/%s.png' % col, bbox_inches='tight')


if __name__ == '__main__':
    __main__()