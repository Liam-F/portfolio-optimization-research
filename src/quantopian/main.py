import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import functools
import random
import pickle
from tqdm import tqdm
from glob import glob
from datetime import timedelta
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, binarize

import features as ft
import preprocessing
import util
from util import plot_confusion_matrix, plot_feature_importance

pd.set_option('display.width', 200)


def filter_pairs(pairs):
    return preprocessing.filter_on_nb_trades(
        preprocessing.filter_on_rolling_sharpe(pairs, 0.8, 252)
    )


def create_inputs(pairs, features_list, end_date=None, drop_nans=True, scale=True):
    if end_date:
        pairs = pairs[:end_date]
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


def create_outputs(pairs):
    nb_strategies = pairs.shape[1]
    y = np.zeros((nb_strategies, 1))
    for i, strategy in enumerate(pairs.columns):
        y[i] = np.array([ft.sharpe_ratio(pairs[strategy])])
    return y


def regression_forest(X, y, features_list, plot=False, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)

    forest = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    forest.fit(X, y)

    te_predictions = forest.predict(X_te).reshape(-1, 1)
    te_real_values = y_te.values.reshape(-1, 1)

    print(f'R2 score: {r2_score(te_real_values, te_predictions)}')

    if plot:
        g = sns.jointplot(te_predictions, te_real_values, kind='reg')
        g.ax_joint.plot([-7, 7], [-7, 7])

    # Classification metrics
    bin_preds = binarize(te_predictions)
    bin_real_values = binarize(te_real_values)

    print(f'F1 score: {f1_score(bin_real_values, bin_preds)}')
    print(f'Confusion matrix: {confusion_matrix(bin_real_values, bin_preds)}')

    if plot:
        plot_feature_importance(features_list, forest)
        plt.show(block=False)

    return forest


def optimal_strategies(complete_pnls, pnls, limit_date=None, n=50):
    if limit_date:
        pnls = pnls[:(limit_date - timedelta(days=1))]
    else:
        limit_date = pnls.index[-1]

    sharpe_ratios_next_month = complete_pnls[limit_date: (limit_date + timedelta(days=30))].apply(ft.sharpe_ratio)
    return sharpe_ratios_next_month.sort_values(ascending=False).index[:n].values


def random_selection(pnls, limit_date=None, n=50):
    if limit_date:
        pnls = pnls[:(limit_date - timedelta(days=1))]
    else:
        limit_date = pnls.index[-1]
    strategies = pnls.columns.values
    random.shuffle(strategies)
    return strategies[:n]


def sharpe_based_selection_function(pnls, limit_date=None, n=50):
    if limit_date:
        pnls = pnls[:(limit_date - timedelta(days=1))]
    sharpes = pnls.apply(ft.sharpe_ratio_last_year)
    return sharpes.sort_values(ascending=False).index[:n].values


def select_n_best_predicted_strategies(model, features_list, pnls, limit_date=None, n=50):
    if limit_date:
        pnls = pnls[:(limit_date - timedelta(days=1))]
    else:
        limit_date = pnls.index[-1]
    precomputed_features_file = f'data/precomputed-features-prod/{limit_date.date()}'
    precomputed_features_file_out = precomputed_features_file + '_output'
    if os.path.exists(precomputed_features_file):
        X = pd.read_csv(precomputed_features_file, parse_dates=True, index_col=0)
    else:
        X = create_inputs(pnls, features_list, scale=True, drop_nans=True)
        X.to_csv(precomputed_features_file)

        y = create_outputs(pnls)
        pd.DataFrame(y, columns=['OUTPUT'], index=pnls.columns).loc[X.index].to_csv(precomputed_features_file_out)
        del y

    strategy_list = X.index.values

    predicted_sharpe = model.predict(X)
    strategy_sharpe_pairs = [(strategy, predicted_sharpe[i])
                             for i, strategy in enumerate(strategy_list)]

    strategy_sharpe_pairs.sort(key=lambda pair: pair[1], reverse=True)

    return [strategy[0] for strategy in strategy_sharpe_pairs[:n]]


def portfolio_selection_simulation(pairs, strategy_selection_fn, start_year, selection_frequency='BM',
                                   change_frequency='BMS'):
    start_date = pairs[start_year:].index[0]
    end_date = pairs[start_date:].index[-1]
    standard_deviations = pairs['2015-02':'2015-12'].std().replace(0, 1)  # exclude 0 standard deviations
    scaled_pairs = pairs / standard_deviations

    selection_dates = pd.date_range(start_date, end_date, freq=selection_frequency)
    change_dates = pd.date_range(start_date, end_date, freq=change_frequency)

    # we want to select the strategies on the first day of operation
    if start_date not in selection_dates:
        selection_dates = selection_dates.union(pd.Index([start_date]))
    if start_date not in change_dates:
        change_dates = change_dates.union(pd.Index([start_date]))

    selection_dates = selection_dates[:len(change_dates)]

    strategy_selection_fn_partial = functools.partial(strategy_selection_fn, pairs)

    with Pool() as p:
        # do the strategy selection over the whole simulation period
        selected_strategies = p.map(strategy_selection_fn_partial, selection_dates)

    selected_strategies_series = pd.Series(data=selected_strategies,
                                           index=change_dates)

    pnls = []
    for date in pairs[start_date:].index:
        if date in change_dates:
            current_strategies = selected_strategies_series.loc[date]
        # record daily pnl
        daily_pnl = scaled_pairs.loc[date, current_strategies].sum()
        pnls.append(daily_pnl)

    return pd.Series(data=pnls, index=pairs[start_date:].index), selected_strategies_series


def compute_training_dataset(features_list, pnls_for_features, pnls_for_labels, features_path=None, scale=True):
    if (features_path is None) or (not os.path.exists(features_path)):
        features = create_inputs(pnls_for_features, features_list, drop_nans=False, scale=False)
        labels = create_outputs(pnls_for_labels)

        training_data = pd.DataFrame(
            data=np.hstack([features, labels]),
            columns=[f.__name__ for f in features_list] + ['OUTPUT'],
            index=pnls_for_features.columns
        ).dropna()
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

    pairs_pnls = pd.read_csv('./data/all-pairs.csv', parse_dates=True, index_col=0)
    pairs_pnls = pairs_pnls['2013':]
    pairs_pnls = preprocessing.filter_on_nb_trades(pairs_pnls,
                                                   percent=0.3)  # filter strategies that have more than 30% of non trading days

    production_strategies_pnls = pd.read_csv('./data/production-strategies.csv', parse_dates=True, index_col=0)

    # min_window_X = 312
    # window_y = 60
    # date_offset = 6
    # training_pairs = production_strategies_pnls['2013':'2015-12']
    # nb_rows = training_pairs.index.shape[0]

    training_data_production = 'data/training_data_monthly.csv'
    # if not os.path.exists(training_data_production):
    #     training_set = []
    #     for i in tqdm(range(min_window_X, nb_rows - window_y, date_offset)):
    #         X_data = training_pairs[:i + 1]
    #         y_data = training_pairs[i + 1:i + window_y + 1]
    #         training_set.append(compute_training_dataset(features_list, X_data, y_data, scale=False))
    #     training_data_unscaled = pd.concat(training_set)
    #     training_data = pd.DataFrame(data=robust_scale(training_data_unscaled),
    #                                         index=training_data_unscaled.index, columns=training_data_unscaled.columns)
    #     training_data.to_csv(training_data_production)
    # else:
    #     training_data = pd.read_csv(training_data_production, parse_dates=True, index_col=0)

    training_data = compute_training_dataset(features_list, pairs_pnls['2013':'2014'],
                                             pairs_pnls['2015-01':'2015-03'],
                                             training_data_file)

    training_features = training_data.drop(['OUTPUT'], axis=1)
    training_labels = training_data['OUTPUT']

    control_pnls, control_selected_strategies = portfolio_selection_simulation(
        production_strategies_pnls['2013':'2016'],
        sharpe_based_selection_function,
        start_year='2016')

    optimal_selection = functools.partial(optimal_strategies, production_strategies_pnls)
    optimal_pnls, optimal_selected_strategies = portfolio_selection_simulation(
        production_strategies_pnls['2013':'2016'],
        optimal_selection,
        start_year='2016'
    )

    forest_sharpes = []
    random_sharpes = []
    for seed in range(100):
        random.seed(seed)

        model = regression_forest(training_features, training_labels, features_list, seed=seed)
        forest_selection_function = functools.partial(select_n_best_predicted_strategies, model, features_list)
        forest_pnls, forest_selected_strategies = portfolio_selection_simulation(
            production_strategies_pnls['2013':'2016'],
            forest_selection_function,
            start_year='2016')

        random_pnls, random_strategies = portfolio_selection_simulation(production_strategies_pnls['2013':'2016'],
                                                                        random_selection,
                                                                        start_year='2016')

        if save_experiment:
            save_experiment_results(forest_pnls, control_pnls, forest_selected_strategies, control_selected_strategies,
                                    experiment_number)

        forest_sharpe = ft.sharpe_ratio(forest_pnls)
        control_sharpe = ft.sharpe_ratio(control_pnls)
        random_sharpe = ft.sharpe_ratio(random_pnls)
        optimal_sharpe = ft.sharpe_ratio(optimal_pnls)

        forest_sharpes.append(forest_sharpe)
        random_sharpes.append(random_sharpe)

        print(f'Sharpe ratio of forest portfolio: {forest_sharpe}')
        print(f'Sharpe ratio of control portfolio: {control_sharpe}')
        print(f'Sharpe ratio of random portfolio: {random_sharpe}')
        print(f'Sharpe ratio of optimal portfolio: {optimal_sharpe}')

        print(
            f'Average number of common elements with forest portfolio: {count_common_elements(optimal_selected_strategies, forest_selected_strategies)}')
        print(
            f'Average number of common elements with control portfolio: {count_common_elements(optimal_selected_strategies, control_selected_strategies)}'
        )
        print(
            f'Average number of common elements with random portfolio: {count_common_elements(optimal_selected_strategies, random_strategies)}'
        )

    forest_sharpes = np.array(forest_sharpes).reshape(-1, 1)
    random_sharpes = np.array(random_sharpes).reshape(-1, 1)
    df = pd.DataFrame(data=np.hstack([forest_sharpes, random_sharpes]), columns=['forest', 'random'])
    df.to_csv('data/noise-test/noise-test-10.csv')
    print(df.describe())

    save_model(model, 'data/simple-forest.pkl')


def count_common_elements(optimal_strategies, selected_strategies):
    total = 0
    for optimal_set, selected_set in zip(optimal_strategies, selected_strategies):
        total += len(set(selected_set).intersection(set(optimal_set)))
    return total / len(optimal_strategies)


if __name__ == '__main__':
    main()
