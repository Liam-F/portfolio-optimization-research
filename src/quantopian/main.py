import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import functools
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale, binarize

import features as ft
import preprocessing as pr
import util
from util import plot_confusion_matrix, plot_feature_importance

pd.set_option('display.width', 200)


def filter_pairs(pairs):
    return pr.filter_on_nb_trades(
        pr.filter_on_rolling_sharpe(pairs, 0.8, 252)
    )


def create_inputs(pairs, features_list, end_date=None, drop_nans=True, scale=True):
    if end_date:
        pairs = pairs[:end_date]
    nb_strategies = pairs.shape[1]
    feature_names = [f.__name__ for f in features_list]

    X = np.zeros((nb_strategies, len(features_list)))
    for i, strategy in enumerate(pairs.columns):
        X[i] = pr.compute_features(pairs[strategy], features_list)
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


def classification_forest(X, y, features_list):
    groups = np.array([-9999, -1, 0, 0.5, 0.75, 1, 9999])
    groups = np.array([-9999, 0.5, 1, 9999])
    labels = np.arange(groups.shape[0] - 1)
    y = pd.cut(y, groups, labels=labels)

    print('Members per Group')
    print(y.value_counts())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    forest = RandomForestClassifier(n_estimators=100, random_state=43, class_weight='balanced_subsample')
    forest.fit(X_tr, y_tr)

    for roc_class in np.arange(0, groups.shape[0] - 1):
        test_predictions = forest.predict(X_te)
        bin_preds = test_predictions
        bin_preds_proba = forest.predict_proba(X_te)[:, roc_class]
        bin_y_te = y_te
        r_fpr, r_tpr, _ = roc_curve((bin_y_te == roc_class) * 1, bin_preds_proba)

        plt.figure()
        plt.title('Class (%s-%s]' % (groups[roc_class], groups[roc_class + 1]))
        r_auc = auc(r_fpr, r_tpr)
        plt.plot(r_fpr, r_tpr, label='AUC: %s' % r_auc)
        plt.legend()
        plt.show(block=False)

    print(f'F1 score: {f1_score(bin_y_te, bin_preds, average="macro")}')

    cm = confusion_matrix(bin_y_te, bin_preds)
    names = ['(%s, %s]' % (a, b) for a, b in zip(groups[:-1], groups[1:])]
    plot_confusion_matrix(cm, names)
    # Useful for comparisons when debugging
    result = y_te.to_frame('true').merge(pd.Series(test_predictions, index=y_te.index, name='pred').to_frame(),
                                         left_index=True, right_index=True)

    plot_feature_importance(features_list, forest)
    plt.show()


def regression_forest(X, y, features_list, plot=False, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)

    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(X_tr, y_tr)

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


def select_n_strategies_sharpe(pnl_pairs, n=100, end_date=None):
    if end_date:
        pnl_pairs = pnl_pairs[:end_date]
    sharpes = pnl_pairs.apply(ft.sharpe_ratio_last_year)
    return sharpes.sort_values(ascending=False).index[:n].values


def select_n_best_predicted_strategies(model, pnl_pairs, features_list, n=100, end_date=None):
    if end_date:
        pnl_pairs = pnl_pairs[:end_date]
    end_date = pnl_pairs.index[-1]
    precomputed_features_file = f'data/precomputed-features/{end_date.date()}'
    if os.path.exists(precomputed_features_file):
        X = pd.read_csv(precomputed_features_file, parse_dates=True, index_col=0)
    else:
        X = create_inputs(pnl_pairs, features_list, scale=True, drop_nans=True)
        X.to_csv(precomputed_features_file)
    strategy_list = X.index.values

    predicted_sharpe = model.predict(X)
    strategy_sharpe_pairs = [(strategy, predicted_sharpe[i])
                             for i, strategy in enumerate(strategy_list)]

    strategy_sharpe_pairs.sort(key=lambda pair: pair[1], reverse=True)

    return [strategy[0] for strategy in strategy_sharpe_pairs[:n]]


def portfolio_selection_simulation(pairs, strategy_selection_fn, start_year='2014', selection_frequency='BM',
                                   change_frequency='BMS'):
    start_date = pairs[start_year:].index[0]
    end_date = pairs[start_date:].index[-1]
    scaled_pairs = pairs / pairs['2015-02':'2015-12'].std()

    selection_dates = pd.date_range(start_date, end_date, freq=selection_frequency)
    change_dates = pd.date_range(start_date, end_date, freq=change_frequency)

    # we want to select the strategies on the first day of operation
    if start_date not in selection_dates:
        selection_dates = selection_dates.union(pd.Index([start_date]))
    if start_date not in change_dates:
        change_dates = change_dates.union(pd.Index([start_date]))

    selection_dates = selection_dates[:len(change_dates)]

    # do the strategy selection over the whole simulation period
    selected_strategies = [strategy_selection_fn(pairs[:selection_date]) for selection_date in selection_dates]
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


def compute_training_dataset(features_path, features_list, pairs, X_tr_date_range, y_tr_date_range):
    if not os.path.exists(features_path):
        print('File %s does not exist, creating and saving' % features_path)

        X = create_inputs(pairs[X_tr_date_range[0]:X_tr_date_range[1]], features_list, drop_nans=False, scale=False)
        y = create_outputs(pairs[y_tr_date_range[0]])

        observations = pd.DataFrame(
            data=np.hstack([X, y]),
            columns=[f.__name__ for f in features_list] + ['OUTPUT'],
            index=pairs.columns
        ).dropna()
        observations = observations[np.all(np.isfinite(observations), axis=1)]

        observations_scaled = pd.DataFrame(
            data=robust_scale(observations),
            columns=observations.columns,
            index=observations.index
        )

        observations_scaled.to_csv(features_path)
    else:
        print('File %s exists!' % features_path)
        observations_scaled = pd.read_csv(features_path, index_col=0)
    return observations_scaled


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


def main():
    experiment_number = 2
    save_experiment = False
    seed_range = 100

    X_tr_date_range = ('2013', '2014')
    y_tr_date_range = ('2015',)

    features_list = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                     ft.skewness, ft.kurtosis, ft.information_ratio,
                     ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                     ft.tail_ratio, ft.common_sense_ratio,
                     ft.sharpe_ratio_last_30_days, ft.sharpe_ratio_last_90_days,
                     ft.sharpe_ratio_last_150_days)

    features_csv = './data/2017-08-15-filtered-pairs-features.csv'

    pairs = pd.read_csv('./data/2017-08-03-filtered-in-sample-pairs.csv', parse_dates=True, index_col=0)

    observations_scaled = compute_training_dataset(features_csv, features_list, pairs, X_tr_date_range,
                                                   y_tr_date_range)

    X = observations_scaled.drop(['OUTPUT'], axis=1)
    y = observations_scaled['OUTPUT']

    sharpes = np.zeros((seed_range, 1))
    control_sharpes = np.zeros((seed_range, 1))
    for seed in range(seed_range):
        forest = regression_forest(X, y, features_list, seed=seed)

        is_pairs = pd.read_csv('data/2017-08-03-in-sample-pairs.csv', parse_dates=True, index_col=0)
        oos_pairs = pd.read_csv('data/2017-08-03-out-sample-pairs.csv', parse_dates=True, index_col=0)
        full_pairs = pd.concat([is_pairs, oos_pairs])
        # full_pairs = full_pairs[full_pairs.columns[:10]]

        control_pnls, control_selected_strategies = portfolio_selection_simulation(full_pairs[:'2016'],
                                                                                   lambda x: select_n_strategies_sharpe(
                                                                                       x,
                                                                                       n=100),
                                                                                   start_year='2016')

        forest_selection = lambda pairs: select_n_best_predicted_strategies(forest, pairs, features_list)
        pnls, selected_strategies = portfolio_selection_simulation(full_pairs[:'2016'], forest_selection,
                                                                   start_year='2016')

        if save_experiment:
            save_experiment_results(pnls, control_pnls, selected_strategies, control_selected_strategies,
                                    experiment_number)

        sharpe = ft.sharpe_ratio(pnls)
        control_sharpe = ft.sharpe_ratio(control_pnls)

        sharpes[seed] = sharpe
        control_sharpes[seed] = control_sharpe

        print(f'Sharpe ratio of forest portfolio: {sharpe}')
        print(f'Sharpe ratio of control portfolio: {control_sharpe}')

    results = pd.DataFrame(data=np.hstack([sharpes, control_sharpes]), columns=['forest_sharpe, control_sharpe'])
    results['difference'] = results['forest_sharpe'] - results['control_sharpe']
    print(results.describe())
    results.to_csv('data/noise-test.csv')


if __name__ == '__main__':
    main()
