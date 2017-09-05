import os
import numpy as np
import pandas as pd
import functools
import pickle
from datetime import timedelta
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor
import features as ft
import preprocessing
import util
from sklearn.linear_model import LinearRegression

pd.set_option('display.width', 200)


def robust_scale(X):
    median = np.median(X, axis=0)
    scale = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    X = X - median
    return X / scale


def compute_features(pairs, features_list, drop_nans=True, scale=True):

    nb_strategies = pairs.shape[1]
    feature_names = [f.__name__ for f in features_list]

    X = np.zeros((nb_strategies, len(features_list)))
    for i, strategy in enumerate(pairs.columns):
        X[i] = preprocessing.compute_features(pairs[strategy], features_list)
    X = pd.DataFrame(data=X, columns=feature_names, index=pairs.columns)
    if drop_nans:
        X = util.drop_nan_rows(X)
    if scale:
        X[:] = robust_scale(X)
    return X


def compute_labels(pairs):
    nb_strategies = pairs.shape[1]
    y = np.zeros((nb_strategies, 1))
    for i, strategy in enumerate(pairs.columns):
        y[i] = np.array([ft.sharpe_ratio(pairs[strategy])])
    return y


def sharpe_based_selection_function(pnls, date, n=50):
    pnls = pnls[:(date - timedelta(days=1))]
    sharpes = pnls.apply(ft.sharpe_ratio_last_year)
    return sharpes.sort_values(ascending=False).index[:n].values


def select_n_best_predicted_strategies(model, features_list, strategy_pnls, date, n=50):
    strategy_pnls = strategy_pnls[:(date - timedelta(days=1))]
    X = compute_features(strategy_pnls, features_list, scale=True, drop_nans=True)

    precomputed_features_file = f'data/precomputed-features-prod/{date.date()}'
    precomputed_features_file_out = precomputed_features_file + '_output'
    if not os.path.exists(precomputed_features_file):
        X.to_csv(precomputed_features_file)

        y = compute_labels(strategy_pnls)
        pd.DataFrame(y, columns=['OUTPUT'], index=strategy_pnls.columns).loc[X.index].to_csv(precomputed_features_file_out)
        del y

    strategy_list = X.index.values
    predicted_sharpe = model.predict(X)
    strategy_sharpe_pairs = [(strategy, predicted_sharpe[i]) for i, strategy in enumerate(strategy_list)]
    strategy_sharpe_pairs.sort(key=lambda pair: pair[1], reverse=True)
    return [strategy[0] for strategy in strategy_sharpe_pairs[:n]]


def portfolio_selection_simulation(strategy_pnls, strategy_selection_function, start_year, selection_frequency='BM', change_frequency='BMS'):
    start_date = strategy_pnls[start_year:].index[0]
    second_date = strategy_pnls[start_year:].index[1]
    end_date = strategy_pnls[start_date:].index[-1]
    scaled_pnls = strategy_pnls / strategy_pnls['2015-02':'2015-12'].std()  # Franky-like scaling (excludes Swiss-Franc event).

    selection_dates = pd.date_range(start_date, end_date, freq=selection_frequency)
    change_dates = pd.date_range(second_date, end_date, freq=change_frequency)

    # we want to select the strategies on the first day of operation
    if start_date not in selection_dates:
        selection_dates = selection_dates.union(pd.Index([start_date]))
    if second_date not in change_dates:
        change_dates = change_dates.union(pd.Index([second_date]))

    selection_dates = selection_dates[:len(change_dates)]

    strategy_selection_fn_partial = functools.partial(strategy_selection_function, strategy_pnls)
    with Pool() as p:
        # do the strategy selection over the whole simulation period
        selected_strategies = p.map(strategy_selection_fn_partial, selection_dates)

    selected_strategies_series = pd.Series(data=selected_strategies,
                                           index=change_dates)

    selected_pnls = []
    for date in strategy_pnls[second_date:].index:
        if date in change_dates:
            current_strategies = selected_strategies_series.loc[date]
        # record daily pnl
        daily_pnl = scaled_pnls.loc[date, current_strategies].sum()
        selected_pnls.append(daily_pnl)

    return pd.Series(data=selected_pnls, index=strategy_pnls[second_date:].index), selected_strategies_series


def compute_training_dataset(features_list, pnls_for_features, pnls_for_labels, features_path=None, scale=True):
    if (features_path is None) or (not os.path.exists(features_path)):
        features = compute_features(pnls_for_features, features_list, drop_nans=False, scale=True)
        labels = compute_labels(pnls_for_labels)

        training_data = pd.DataFrame(
            data=np.hstack([features, labels]),
            columns=[f.__name__ for f in features_list] + ['OUTPUT'],
            index=pnls_for_features.columns
        ).dropna()

        if features_path:
            training_data.to_csv(features_path.replace('.csv', '_unscaled.csv'))

        training_data = training_data[np.all(np.isfinite(training_data), axis=1)]

        if scale:
            training_data_final = pd.DataFrame(
                data=robust_scale(training_data),
                columns=training_data.columns,
                index=training_data.index
            )
        else:
            training_data_final = training_data

        if features_path:
            training_data_final.to_csv(features_path)
    else:
        training_data_final = pd.read_csv(features_path, index_col=0)
    return training_data_final


def save_experiment_results(pnls, control_pnls, selected_strategies, control_selected_strategies, experiment_number):
    experiment_number_suffix = '{:02d}'.format(experiment_number)
    experiment_dir_path = f'data/experiment-{experiment_number_suffix}'
    if not os.path.exists(experiment_dir_path):
        os.mkdir(experiment_dir_path)
    selected_strategies.to_csv(
        f'{experiment_dir_path}/selected_strategies-experiment-{experiment_number_suffix}-forest.csv')
    control_selected_strategies.to_csv(
        f'{experiment_dir_path}/selected_strategies-experiment-{experiment_number_suffix}-control.csv')
    pnls.to_csv(f'{experiment_dir_path}/pnls-experiment-{experiment_number_suffix}-forest.csv')
    control_pnls.to_csv(f'{experiment_dir_path}/pnls-experiment-{experiment_number_suffix}-control.csv')


def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def main():
    experiment_number = 3
    save_experiment = True

    features_list = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                     ft.skewness, ft.kurtosis, ft.information_ratio,
                     ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                     ft.tail_ratio, ft.common_sense_ratio,
                     ft.sharpe_ratio_last_30_days, ft.sharpe_ratio_last_90_days,
                     ft.sharpe_ratio_last_150_days)

    training_data_file = './data/training_data.csv'
    strategy_pnls = pd.read_csv('./data/all-pairs.csv', parse_dates=True, index_col=0)
    # training_data_file = './data/training_data-all_ll.csv'
    # strategy_pnls = pd.read_csv('./data/all-pairs_ll-pairs.csv', parse_dates=True, index_col=0)
    strategy_pnls = strategy_pnls['2013':'2016']
    strategy_pnls = preprocessing.filter_on_nb_trades(strategy_pnls, percent=0.3)  # filter strategies that have more than 30% of non trading days
    print('Remaining number of strategy PNLs: %s' % strategy_pnls.shape[1])

    production_strategies_pnls = pd.read_csv('./data/production-strategies.csv', parse_dates=True, index_col=0)
    production_strategies_pnls = production_strategies_pnls['2013':'2016']

    training_data = compute_training_dataset(features_list, strategy_pnls['2013':'2014'], strategy_pnls['2015'],
                                             training_data_file)

    training_features = training_data.drop(['OUTPUT'], axis=1)
    training_labels = training_data['OUTPUT']

    control_pnls, control_selected_strategies = portfolio_selection_simulation(production_strategies_pnls['2013':'2016'],
                                                                               sharpe_based_selection_function,
                                                                               start_year='2016')

    from sklearn.feature_selection import RFE
    from sklearn.cross_decomposition import PLSRegression
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # lm_model = LinearRegression()
    # pls = PLSRegression(n_components=13)
    #
    # model = pls
    model.fit(training_features, training_labels)

    # rfe = RFE(pls, n_features_to_select=1, step=1, verbose=1)
    # model = rfe
    # model.fit(training_features, training_labels)
    # lm_model.fit(training_features, training_labels)
    # model = lm_model

    import statsmodels.api as sm
    lm = sm.OLS(training_labels, sm.add_constant(training_features)).fit()
    print(lm.summary())

    forest_selection_function = functools.partial(select_n_best_predicted_strategies, model, features_list)
    forest_pnls, forest_selected_strategies = portfolio_selection_simulation(production_strategies_pnls['2013':'2016'],
                                                                             forest_selection_function,
                                                                             start_year='2016')

    if save_experiment:
        save_experiment_results(forest_pnls, control_pnls, forest_selected_strategies, control_selected_strategies,
                                experiment_number)

    forest_sharpe = ft.sharpe_ratio(forest_pnls)
    control_sharpe = ft.sharpe_ratio(control_pnls)

    print(f'Sharpe ratio of forest portfolio: {forest_sharpe}')
    print(f'Sharpe ratio of control portfolio: {control_sharpe}')

    pairs_to_check = '2016-08-31'
    X = pd.read_csv('./data/precomputed-features-prod/%s' % pairs_to_check, parse_dates=True, index_col=0)
    y = pd.read_csv('./data/precomputed-features-prod/%s_output' % pairs_to_check, parse_dates=True, index_col=0)

    print('Plotting boundaries!')
    make_estimation_boundary_plots(X, y, model, block_at_end=True, standardize=False)

    save_model(model, 'data/simple-forest.pkl')


def make_estimation_boundary_plots(X, y, model, block_at_end=True, lines_per_plot=20, abort_after=200,
                                   standardize=False):
    return
    import matplotlib.pyplot as plt

    forest_pred = model.predict(X)

    result = np.zeros((model.n_estimators, X.shape[0]))
    for i, tree in enumerate(model.estimators_):
        result[i, :] = tree.predict(X)
    trees_predicts = result.mean(axis=0)
    trees_std = result.std(axis=0)

    for i in range(trees_predicts.shape[0]):
        if i > abort_after:
            break

        if i % lines_per_plot == 0:
            plt.figure()

        pred = trees_predicts[i]
        std = trees_std[i]
        act = y['OUTPUT'].iloc[i]

        left = pred - 1.96 * std / np.sqrt(model.n_estimators)
        right = pred + 1.96 * std / np.sqrt(model.n_estimators)

        plt.plot(pred, i, 'ro', c='r')
        plt.plot([left, right], [i, i], c='r')
        plt.plot(act, i, 'ro', c='g')

        if i % lines_per_plot == lines_per_plot - 1:
            plt.tight_layout()
            plt.show(block=False)

    xs = np.argsort(np.argsort(forest_pred))
    ys = np.argsort(np.argsort(y['OUTPUT'].values))

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel('Predicted OOS Sharpe Ranking')
    plt.ylabel('Actual OOS Sharpe Ranking')
    plt.show(block=False)

    xs = forest_pred
    ys = y['OUTPUT'].values

    if standardize:
        xs = (xs - xs.mean()) / xs.std()
        ys = (ys - ys.mean()) / ys.std()

    plt.figure()
    plt.scatter(xs, ys)
    suffix = ' (std)' if standardize else ''
    plt.xlabel('Predicted OOS Sharpe' + suffix)
    plt.ylabel('Actual OOS Sharpe' + suffix)
    plt.show(block=False)

    import statsmodels.api as sm
    print(sm.OLS(xs, sm.add_constant(ys)).fit().summary())

    plt.show(block=block_at_end)

    return

if __name__ == '__main__':
    main()
