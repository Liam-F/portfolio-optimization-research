import functools

import pandas as pd

from quantopian.main import portfolio_selection_simulation, sharpe_based_selection_function, \
    select_n_best_predicted_strategies

import features as ft

from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import resample

if __name__ == '__main__':
    training_data = pd.read_csv('./data/training_data.csv', index_col=0)

    training_features = training_data.drop(['OUTPUT'], axis=1)
    training_labels = training_data['OUTPUT']

    pls_model = PLSRegression(n_components=3)
    pls_model.fit(training_features, training_labels)

    features_list = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                     ft.skewness, ft.kurtosis, ft.information_ratio,
                     ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                     ft.tail_ratio, ft.common_sense_ratio,
                     ft.sharpe_ratio_last_30_days, ft.sharpe_ratio_last_90_days,
                     ft.sharpe_ratio_last_150_days)

    strategy_pnls = pd.read_csv('./data/all-pairs.csv', parse_dates=True, index_col=0)
    strategy_pnls = strategy_pnls['2013':'2016']

    results = []

    for i in range(100):
        subset = resample(training_data, n_samples=1000, replace=True)
        subset_pnls = strategy_pnls[subset.index]

        model_selection_function = functools.partial(select_n_best_predicted_strategies, pls_model, features_list)
        model_pnls, model_selected_strategies = portfolio_selection_simulation(
            subset_pnls['2013':'2016'], model_selection_function, start_year='2016'
        )

        control_pnls, control_selected_strategies = portfolio_selection_simulation(
            subset_pnls['2013':'2016'], sharpe_based_selection_function, start_year='2016'
        )

        model_sharpe = ft.sharpe_ratio(model_pnls)
        control_sharpe = ft.sharpe_ratio(control_pnls)

        results.append([i, model_sharpe, control_sharpe])

        print('[%4d] Control: %2.2f, PLS: %2.2f' % (i, control_sharpe, model_sharpe))

    results = pd.DataFrame(results, columns='iter model control'.split()).set_index('iter')
    results.to_clipboard()
    print(results)