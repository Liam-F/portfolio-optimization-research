import datetime
import functools
import os
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import quantopian.features as ft
from quantopian.RobustScaler import RobustScaler
from quantopian.main import compute_features, compute_labels

CACHE_FILES_FOLDER = './data/bootstrap_simulation_cached_features_prod/'
sns.set()


def get_simulation_dates(strategy_pnls, start_year='2016', selection_frequency='BM', change_frequency='BMS'):
    start_date = strategy_pnls[start_year:].index[0]
    second_date = strategy_pnls[start_year:].index[1]
    end_date = strategy_pnls[start_date:].index[-1]

    selection_dates = pd.date_range(start_date, end_date, freq=selection_frequency)
    change_dates = pd.date_range(second_date, end_date, freq=change_frequency)

    # we want to select the strategies on the first day of operation
    if start_date not in selection_dates:
        selection_dates = selection_dates.union(pd.Index([start_date]))
    if second_date not in change_dates:
        change_dates = change_dates.union(pd.Index([second_date]))

    selection_dates = selection_dates[:len(change_dates)]

    print('Change Dates: %s' % change_dates)
    print('Selection Dates: %s' % selection_dates)

    return change_dates, selection_dates


def compute_cache_files_date(pnls, features, date, cache_files_folder=CACHE_FILES_FOLDER):
    if hasattr(date, 'date'):
        date = date.date()
    print('Computing cached features file for %s' % date)

    pnls_expanding = pnls[:(date - datetime.timedelta(days=1))]

    observations = compute_features(pnls_expanding, features, drop_nans=True, scale=False)
    observations = observations.dropna()

    fn = cache_files_folder + '%s.csv' % date
    observations.to_csv(fn)
    print('Saved %s' % fn)


def compute_cache_files(pnls, dates, features, cache_files_folder=CACHE_FILES_FOLDER, pool_size=None):
    func = functools.partial(compute_cache_files_date, pnls, features, cache_files_folder=cache_files_folder)

    with Pool(pool_size) as p:
        p.map(func, dates)


def get_features_and_labels(pnls, features, X_window, y_window, force_recompute=False,
                            cache_files_folder=CACHE_FILES_FOLDER):
    feature_pnls = pnls[X_window[0]:X_window[1]]
    label_pnls = pnls[y_window[0]:y_window[1]]
    last_date = feature_pnls.index[-1]

    if hasattr(last_date, 'date'):
        last_date = last_date.date()

    fn = cache_files_folder + '%s.csv' % last_date

    if force_recompute or not os.path.exists(fn):
        print('Recomputing features...')
        compute_cache_files(pnls, [last_date], features, cache_files_folder=cache_files_folder)

    X = pd.read_csv(fn, index_col=0)
    y = compute_labels(label_pnls, return_series=True)
    y = y.loc[X.index]

    return X, y


def load_features(pnls, date, cache_files_folder=CACHE_FILES_FOLDER):
    if hasattr(date, 'date'):
        date = date.date()

    fn = cache_files_folder + '%s.csv' % date

    if not os.path.exists(fn):
        raise Exception('Path %s with features did not exist. Did you pre-compute them?' % fn)

    X = pd.read_csv(fn, index_col=0)
    X = X.loc[pnls.columns].dropna()

    return X


def build_linear_model(pnls, features, X_window=('2013', '2014'), y_window=('2015', '2015'), force_recompute=False,
                       cache_files_folder=CACHE_FILES_FOLDER):
    X, y = get_features_and_labels(pnls, features, X_window, y_window, force_recompute=force_recompute,
                                   cache_files_folder=cache_files_folder)

    tmp = X.merge(y.to_frame('__OUTPUT__'), left_index=True, right_index=True).dropna()
    X = tmp.drop('__OUTPUT__', axis=1)
    y = tmp['__OUTPUT__'].rename('y_true')

    robust_scaler = RobustScaler()
    pls = LinearRegression()
    pipe = Pipeline(steps=[('scaler', robust_scaler), ('pls', pls)])
    pipe.fit(X, y)

    return pipe


def build_pls_model(pnls, features, X_window=('2013', '2014'), y_window=('2015', '2015'), force_recompute=False,
                    cache_files_folder=CACHE_FILES_FOLDER):
    X, y = get_features_and_labels(pnls, features, X_window, y_window, force_recompute=force_recompute,
                                   cache_files_folder=cache_files_folder)

    tmp = X.merge(y.to_frame('__OUTPUT__'), left_index=True, right_index=True).dropna()
    X = tmp.drop('__OUTPUT__', axis=1)
    y = tmp['__OUTPUT__'].rename('y_true')

    robust_scaler = RobustScaler()
    pls = PLSRegression(n_components=3)
    pipe = Pipeline(steps=[('scaler', robust_scaler), ('pls', pls)])
    pipe.fit(X, y)

    return pipe


def build_rf_model(pnls, features, X_window=('2013', '2014'), y_window=('2015', '2015'), force_recompute=False,
                   cache_files_folder=CACHE_FILES_FOLDER):
    X, y = get_features_and_labels(pnls, features, X_window, y_window, force_recompute=force_recompute,
                                   cache_files_folder=cache_files_folder)

    tmp = X.merge(y.to_frame('__OUTPUT__'), left_index=True, right_index=True).dropna()
    X = tmp.drop('__OUTPUT__', axis=1)
    y = tmp['__OUTPUT__'].rename('y_true')

    robust_scaler = RobustScaler()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    pipe = Pipeline(steps=[('scaler', robust_scaler), ('rf', rf)])
    pipe.fit(X, y)

    return pipe


def build_portfolios(pnls, models, selection_dates, change_dates=None, portfolio_size=100,
                     control_feature='sharpe_ratio_last_year'):
    model_portfolios = [pd.DataFrame(0, index=selection_dates, columns=pnls.columns) for _ in models]
    control_portfolio = pd.DataFrame(0, index=selection_dates, columns=pnls.columns)

    for date in selection_dates:
        X = load_features(pnls, date)
        control_strategies = X[control_feature].sort_values(ascending=False).index[:portfolio_size]
        control_portfolio.loc[date, control_strategies] = 1

        for model, model_portfolio in zip(models, model_portfolios):
            y_pred = pd.Series(model.predict(X).reshape(-1), index=X.index)
            model_strategies = y_pred.sort_values(ascending=False).index[:portfolio_size]
            model_portfolio.loc[date, model_strategies] = 1

    if change_dates is not None:
        control_portfolio.set_index(change_dates)

        for model_portfolio in model_portfolios:
            model_portfolio.set_index(change_dates)

    return [control_portfolio] + model_portfolios


def compute_pnls(pnls, rebalance_dates, portfolios, portfolio_names=None, first_portfolio_name=None):
    if portfolio_names is None:
        portfolio_names = ['Portfolio %s' % i for i in range(1, len(portfolios) + 1)]
    if first_portfolio_name is not None:
        portfolio_names[0] = first_portfolio_name

    portfolios_index = pnls[rebalance_dates[0]:].index
    portfolio_pnls = pd.DataFrame(np.nan, index=portfolios_index, columns=range(len(portfolios)))

    for i, portfolio in enumerate(portfolios):
        upsampled_portfolio = portfolio.reindex(index=portfolio_pnls.index, method='ffill')
        portfolio_pnls[i] = pnls.mul(upsampled_portfolio, fill_value=0).sum(axis=1)

    portfolio_pnls.columns = portfolio_names

    return portfolio_pnls


def do_bootstrap(pnls, models, selection_dates, change_dates, portfolio_names, subsample_size=None, portfolio_size=100):
    if subsample_size is not None:
        pnls = pnls.sample(subsample_size, replace=False, axis=1)

    portfolios = build_portfolios(pnls, models, selection_dates, change_dates,
                                                           portfolio_size=portfolio_size)
    portfolio_pnls = compute_pnls(pnls, change_dates, portfolios, portfolio_names)

    sharpes = portfolio_pnls.mean() / portfolio_pnls.std()

    return sharpes.values


def do_bootstrap_ignored_arg(pnls, models, selection_dates, change_dates, portfolio_names, i, subsample_size=None,
                             portfolio_size=100):
    return do_bootstrap(pnls, models, selection_dates, change_dates, portfolio_names, subsample_size=subsample_size,
                        portfolio_size=portfolio_size)


def main():
    all_pair_pnls_train = pd.read_csv('./data/all-pairs.csv', parse_dates=True, index_col=0)
    all_pair_pnls_train = all_pair_pnls_train['2014':'2016']

    production_strategies_pnls = pd.read_csv('./data/production-strategies.csv', parse_dates=True, index_col=0)
    production_strategies_pnls = production_strategies_pnls['2015':'2017']

    all_pair_pnls = production_strategies_pnls

    change_dates, selection_dates = get_simulation_dates(all_pair_pnls, start_year='2017')

    features = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                ft.skewness, ft.kurtosis, ft.information_ratio,
                ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                ft.tail_ratio, ft.common_sense_ratio,
                ft.sharpe_ratio_last_30_days, ft.sharpe_ratio_last_90_days,
                ft.sharpe_ratio_last_150_days)

    force_recompute = True
    pool_size = 5

    print('Computing Cache Files (Forced: %s)' % force_recompute)
    if force_recompute:
        compute_cache_files(all_pair_pnls, selection_dates, features, pool_size=pool_size)
        force_recompute = False

    print('Building Models')
    CACHE_FILES_FOLDER_TRAIN = './data/bootstrap_simulation_cached_features/'
    models, model_names = get_selected_models(all_pair_pnls_train, CACHE_FILES_FOLDER_TRAIN, force_recompute, features)

    scaled_pnls = all_pair_pnls / all_pair_pnls['2015-02':'2015-12'].std()
    scaled_pnls = scaled_pnls.dropna(axis=1)

    print('Running Bootstrap')
    func = functools.partial(do_bootstrap_ignored_arg, scaled_pnls, models, selection_dates,
                             change_dates, model_names, subsample_size=100, portfolio_size=50)
    with Pool(pool_size) as pool:
        results = pool.map(func, tqdm.tqdm(range(1000)))

    results = np.sqrt(252) * pd.DataFrame(results, columns=model_names)

    ax = results.boxplot()
    ax.set_ylabel('Out Of Sample Sharpe')

    ax = results.plot(kind='hist', bins=25, alpha=.5)
    ax.set_xlabel('Out Of Sample Sharpe')
    plt.show()

    pass


def get_linear_models(pnls, cache_folder, force_recompute, features):
    models = [
        build_linear_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-12'),
                           cache_files_folder=cache_folder),
        build_linear_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-10'),
                           cache_files_folder=cache_folder),
        build_linear_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-08'),
                           cache_files_folder=cache_folder),
        build_linear_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-06'),
                           cache_files_folder=cache_folder),
        build_linear_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-04'),
                           cache_files_folder=cache_folder),
        build_linear_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-02'),
                           cache_files_folder=cache_folder)
    ]

    model_names = [
        'Control', 'LM 12M', 'LM 10M', 'LM 8M', 'LM 6M', 'LM 4M', 'LM 2M'
    ]

    return models, model_names


def get_pls_models(pnls, cache_folder, force_recompute, features):
    models = [
        build_pls_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-12'),
                        cache_files_folder=cache_folder),
        build_pls_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-10'),
                        cache_files_folder=cache_folder),
        build_pls_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-08'),
                        cache_files_folder=cache_folder),
        build_pls_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-06'),
                        cache_files_folder=cache_folder),
        build_pls_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-04'),
                        cache_files_folder=cache_folder),
        build_pls_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-02'),
                        cache_files_folder=cache_folder)
    ]

    model_names = [
        'Control', 'PLS 12M', 'PLS 10M', 'PLS 8M', 'PLS 6M', 'PLS 4M', 'PLS 2M'
    ]

    return models, model_names


def get_rf_models(pnls, cache_folder, force_recompute, features):
    models = [
        build_rf_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-12'),
                       cache_files_folder=cache_folder),
        build_rf_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-10'),
                       cache_files_folder=cache_folder),
        build_rf_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-08'),
                       cache_files_folder=cache_folder),
        build_rf_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-06'),
                       cache_files_folder=cache_folder),
        build_rf_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-04'),
                       cache_files_folder=cache_folder),
        build_rf_model(pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-02'),
                       cache_files_folder=cache_folder)
    ]

    model_names = [
        'Control', 'RF 12M', 'RF 10M', 'RF 8M', 'RF 6M', 'RF 4M', 'RF 2M'
    ]

    return models, model_names


def get_selected_models(pnls, cache_folder, force_recompute, features):
    models = [
        build_pls_model(pnls, features, force_recompute=force_recompute,
                        X_window=('2014-01', '2015-12'), y_window=('2016-01', '2016-12'),
                        cache_files_folder=cache_folder),
        build_rf_model(pnls, features, force_recompute=force_recompute,
                       X_window=('2014-01', '2015-12'), y_window=('2016-01', '2016-12'),
                       cache_files_folder=cache_folder)
    ]

    model_names = ['Control', 'PLS 12M', 'RF 12M']

    return models, model_names


if __name__ == '__main__':
    main()


# pls_model = build_pls_model(all_pair_pnls_train, features, force_recompute=force_recompute,
#                             cache_files_folder=CACHE_FILES_FOLDER_TRAIN)
# pls_model_3_months = build_pls_model(all_pair_pnls_train, features, force_recompute=force_recompute,
#                                      y_window=('2015-01', '2015-03'), cache_files_folder=CACHE_FILES_FOLDER_TRAIN)
# pls_model_1_month = build_pls_model(all_pair_pnls_train, features, force_recompute=force_recompute,
#                                     y_window=('2015-01', '2015-01'), cache_files_folder=CACHE_FILES_FOLDER_TRAIN)
# rf_model = build_rf_model(all_pair_pnls_train, features, force_recompute=force_recompute,
#                           cache_files_folder=CACHE_FILES_FOLDER_TRAIN)
# rf_model_3_months = build_rf_model(all_pair_pnls_train, features, force_recompute=force_recompute,
#                                    y_window=('2015-01', '2015-03'), cache_files_folder=CACHE_FILES_FOLDER_TRAIN)
# rf_model_1_month = build_rf_model(all_pair_pnls_train, features, force_recompute=force_recompute,
#                                   y_window=('2015-01', '2015-01'), cache_files_folder=CACHE_FILES_FOLDER_TRAIN)
# models = [pls_model, pls_model_3_months, pls_model_1_month, rf_model, rf_model_3_months, rf_model_1_month]
# model_names = ['Control',
#                'PLS (1 Year)', 'PLS (3 Months)', 'PLS (1 Month)',
#                'RF (1 Year)', 'RF (3 Months)', 'RF (1 Month)']

# pls_model_11_months = build_pls_model(all_pair_pnls, features, force_recompute=force_recompute,
#                                       y_window=('2015-01', '2015-11'))
# rf_model_11_months = build_rf_model(all_pair_pnls, features, force_recompute=force_recompute,
#                                     y_window=('2015-01', '2015-11'))
# models = [pls_model_11_months, rf_model_11_months]
# model_names = ['Control', 'PLS (11 Months)', 'RF (11 Months)']

# models = [
#     build_rf_model(all_pair_pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-12')),
#     build_rf_model(all_pair_pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-10')),
#     build_rf_model(all_pair_pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-08')),
#     build_rf_model(all_pair_pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-06')),
#     build_rf_model(all_pair_pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-04')),
#     build_rf_model(all_pair_pnls, features, force_recompute=force_recompute, y_window=('2015-01', '2015-02'))
# ]
#
# model_names = [
#     'Control', 'RF 12M', 'RF 10M', 'RF 8M', 'RF 6M', 'RF 4M', 'RF 2M'
# ]