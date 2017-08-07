import numpy as np
import pandas as pd
import preprocessing as pr
import features as ft
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import robust_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score, confusion_matrix
from sklearn.preprocessing import binarize

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
        y[i] = np.array([ft.sharpe_ratio(pairs['2015':'2015'][strategy])])
    return y


def main():
    features_list = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                     ft.skewness, ft.kurtosis, ft.information_ratio,
                     ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                     ft.tail_ratio, ft.common_sense_ratio)

    if not os.path.exists('./data/2017-08-03-filtered-pairs-features.csv'):
        print('File does not exist, creating and saving')
        pairs = pd.read_csv('./data/2017-08-03-filtered-in-sample-pairs.csv', parse_dates=True, index_col=0)
        # pairs = pairs[pairs.columns[:1000]]

        X = create_inputs(pairs, features_list)
        y = create_outputs(pairs, features_list)

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

        observations_scaled.to_csv('./data/2017-08-03-filtered-pairs-features.csv')
    else:
        print('File exists!')
        observations_scaled = pd.read_csv('./data/2017-08-03-filtered-pairs-features.csv', index_col=0)

    # Throw out observations which are more than 4 standard deviations away from the mean
    # Note: doesn't really matter for Random Forests
    # observations_scaled = observations_scaled[np.all(observations_scaled.abs() <= 4, axis=1)]

    # sns.pairplot(observations_scaled)
    # plt.savefig('pairplot.png', bbox_inches='tight')

    X = observations_scaled.drop(['OUTPUT'], axis=1)
    y = observations_scaled['OUTPUT']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    forest = RandomForestRegressor(n_estimators=100, random_state=43)#, criterion='mae', min_samples_leaf=8, min_samples_split=16)
    forest.fit(X_tr, y_tr)

    test_predictions = forest.predict(X_te)
    te_size = test_predictions.shape[0]
    predictions = pd.DataFrame(
        data=np.hstack([test_predictions.reshape(te_size, 1), y_te.values.reshape(te_size, 1)]),
        columns=['predicted Sharpe', 'real Sharpe']
    )
    sns.pairplot(predictions)

    # plt.figure()
    # sns.jointplot(pd.Series(test_predictions, name='Pred'), y_te.rename('True'), kind='reg')

    print(f'R2 score: {r2_score(y_te, test_predictions)}')

    bin_preds = binarize(test_predictions.reshape(-1, 1))
    bin_y_te = binarize(y_te).reshape(-1, 1)
    print(f'F1 score: {f1_score(bin_y_te, bin_preds)}')
    print(f'Confusion matrix: {confusion_matrix(bin_y_te, bin_preds)}')

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    features_names = [features_list[i].__name__ for i in indices]
    plt.figure()
    plt.title("Feature importance")
    plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], align='center')
    plt.xticks(range(X.shape[1]), features_names, rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()