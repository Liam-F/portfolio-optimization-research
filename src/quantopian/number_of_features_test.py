import pandas as pd
from sklearn.cross_decomposition import PLSRegression

import quantopian.features as ft

from quantopian.main import compute_features, compute_labels
from quantopian.RobustScaler import RobustScaler

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def precompute_everything(pnls, features):
    X = compute_features(pnls['2013':'2014'], features, drop_nans=True, scale=True)
    y = compute_labels(pnls['2015':'2015'], return_series=True).rename('OOS Sharpe')
    Xy = X.merge(y.to_frame(), left_index=True, right_index=True, how='left')

    Xy.to_csv('data/number_of_features_test_13_14_15.csv')

    X = compute_features(pnls['2014':'2015'], features, drop_nans=True, scale=True)
    y = compute_labels(pnls['2016':'2016'], return_series=True).rename('OOS Sharpe')
    Xy = X.merge(y.to_frame(), left_index=True, right_index=True, how='left')

    Xy.to_csv('data/number_of_features_test_14_15_16.csv')


def build_rf_model(X, y):
    robust_scaler = RobustScaler()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    pipe = Pipeline(steps=[('scaler', robust_scaler), ('rf', rf)])
    pipe.fit(X, y)

    return pipe


def build_pls_model(X, y):
    robust_scaler = RobustScaler()

    n_components = min(X.shape[1], 3)

    pls = PLSRegression(n_components=n_components)
    pipe = Pipeline(steps=[('scaler', robust_scaler), ('pls', pls)])
    pipe.fit(X, y)

    return pipe


def __main__():
    pnls = pd.read_csv('data/all-pairs.csv', parse_dates=True, index_col=0)

    features = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                ft.skewness, ft.kurtosis, ft.information_ratio,
                ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                ft.tail_ratio, ft.common_sense_ratio,
                ft.sharpe_ratio_last_30_days, ft.sharpe_ratio_last_90_days,
                ft.sharpe_ratio_last_150_days)

    # precompute_everything(pnls, features)

    in_sample = pd.read_csv('data/number_of_features_test_13_14_15.csv', index_col=0).dropna()
    in_sample_X = in_sample.drop('OOS Sharpe', axis=1)
    in_sample_y = in_sample['OOS Sharpe']

    val_sample = pd.read_csv('data/number_of_features_test_14_15_16.csv', index_col=0).dropna()
    val_sample_X = val_sample.drop('OOS Sharpe', axis=1)
    val_sample_y = val_sample['OOS Sharpe']

    scoring_function = metrics.mean_squared_error

    selected_features = []
    scores = []
    for i in range(1, in_sample_X.shape[1] + 1):
        print('[%2d] Starting iteration.' % i)
        print('[%2d] Preselected Features: %s' % (i, ', '.join(selected_features)))

        available_features = set(in_sample_X.columns) - set(selected_features)
        feature_scores = []
        for feature in available_features:
            print('[%2d]    Checking Feature %30s... ' % (i, feature), end='')
            features = selected_features + [feature]
            is_X, is_y = in_sample_X[features], in_sample_y
            va_X, va_y = val_sample_X[features], val_sample_y

            model = build_pls_model(is_X, is_y)

            is_y_pred = model.predict(is_X)
            va_y_pred = model.predict(va_X)

            is_score = scoring_function(is_y, is_y_pred)
            va_score = scoring_function(va_y, va_y_pred)

            feature_scores.append((feature, is_score, va_score))

            print('In: %5f, Out: %5f' % (is_score, va_score))

        optimal_feature, is_score, va_score = min(feature_scores, key=lambda trip: trip[1])
        selected_features = selected_features + [optimal_feature]
        scores.append((optimal_feature, is_score, va_score))
        print('[%2d] Iteration Done. Best feature set: %s' % (i, ', '.join(selected_features)))

    result = pd.DataFrame(scores, columns=['Optimal Feature', 'In Sample', 'Validation'],
                          index=range(1, is_X.shape[1] + 1))
    print(result)

if __name__ == '__main__':
    __main__()