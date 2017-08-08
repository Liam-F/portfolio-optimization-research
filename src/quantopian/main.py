import numpy as np
import pandas as pd
import preprocessing as pr
import features as ft
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import robust_scale, binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, r2_score

import tqdm

pd.set_option('display.width', 200)
sns.set_style('white')


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


def create_outputs(pairs):
    nb_strategies = pairs.shape[1]
    y = np.zeros((nb_strategies, 1))
    for i, strategy in enumerate(pairs.columns):
        y[i] = np.array([ft.sharpe_ratio(pairs['2015':'2015'][strategy])])
    return y


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    confusion = pd.DataFrame(cm, index=classes, columns=classes)
    print(confusion)

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Sharpe (Out of Sample)')
    plt.xlabel('Predicted Sharpe (In Sample)')


def plot_feature_importance(features_list, forest):
    """
    This feature plots the features of a random forest ordered by their importance
    """
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    features_names = [features_list[i].__name__ for i in range(len(importances))]
    feature_importances = pd.Series(importances, features_names).rename(index={
        'annret': 'Annualized Return',
        'annvol': 'Annualized Volatility',
        'information_ratio': 'Information Ratio',
        'tail_ratio': 'Tail Ratio',
        'skewness': 'Skewness',
        'common_sense_ratio': 'Common Sense Ratio',
        'calmar_ratio': 'Calmar Ratio',
        'max_drawdown': 'Max Drawdown',
        'sharpe_ratio': 'In-Sample Sharpe Ratio',
        'drawdown_area': 'Drawdown Area',
        'sharpe_std': 'STD of In-Sample Sharpe Ratio',
        'sharpe_ratio_last_year': '12M Sharpe Ratio',
        'sharpe_ratio_last_30_days': '1M Sharpe Ratio',
        'sharpe_ratio_last_90_days': '3M Sharpe Ratio',
        'sharpe_ratio_last_150_days': '5M Sharpe Ratio',
        'sortino_ratio': 'Sortino',
        'trading_days': 'Trading Days (PnL != 0)',
        'kurtosis': 'Kurtosis'
    })
    plt.figure()
    feature_importances.sort_values().plot(kind='barh', xerr=std, title='Feature Importance')
    plt.tight_layout()


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


def regression_forest(X, y, features_list):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(X_tr, y_tr)

    te_predictions = forest.predict(X_te).reshape(-1, 1)
    te_real_values = y_te.values.reshape(-1, 1)

    print(f'R2 score: {r2_score(te_real_values, te_predictions)}')

    g = sns.jointplot(te_predictions, te_real_values, kind='reg')
    g.ax_joint.plot([-7, 7], [-7, 7])

    # Classification metrics
    bin_preds = binarize(te_predictions)
    bin_real_values = binarize(te_real_values)

    print(f'F1 score: {f1_score(bin_real_values, bin_preds)}')
    print(f'Confusion matrix: {confusion_matrix(bin_real_values, bin_preds)}')

    plot_feature_importance(features_list, forest)
    plt.show()

    return forest


def select_n_best_predicted_strategies(model, pnl_pairs, features_list, n=100):
    strategy_list = pnl_pairs.columns
    X = create_inputs(pnl_pairs, features_list)

    predicted_sharpe = model.predict(X)
    strategy_sharpe_pairs = [(strategy, predicted_sharpe[i])
                             for i, strategy in enumerate(strategy_list)]

    strategy_sharpe_pairs.sort(key=lambda pair: pair[1], reverse=True)

    return strategy_sharpe_pairs[:n]


def optimize(features, pairs, normalize=True, standardize=False, start=252*2, frequency=5):
    dates = pairs.index

    result = []

    orig_pairs = pairs.copy()

    if standardize and normalize:
        raise Exception("Make up your mind. Standardize or normalize (natural language or).")

    # Should do it in the loop, but takes forever -.-'
    if normalize:
        pairs = pairs.copy().apply(lambda x: x / (np.max(x) - np.min(x)))

    if standardize:
        pairs = (pairs - pairs.mean()) / pairs.std()

    print('Running portfolio')
    for i in tqdm.tqdm(np.arange(start, dates.shape[0] - 1, frequency)):

        train_set = pairs[i - start:i]
        sharpes = train_set.mean() / train_set.std()

        # if normalize:
        #     train_set = train_set.apply(lambda x: x / (np.max(x) - np.min(x)))
        #
        # if standardize:
        #     train_set = (train_set - train_set.mean()) / train_set.std()

        selected_strategies = sharpes.sort_values(ascending=False).index[:20]

        for j in range(i, i + frequency):
            if j >= dates.shape[0] - 1:
                continue

            tomorrow = dates[j + 1]  # Select portfolio today, hold it tomorrow

            pnl = orig_pairs.loc[tomorrow, selected_strategies].sum()

            result.append([tomorrow, pnl])

    result = pd.DataFrame(result, columns=['Date', 'PnL']).set_index('Date')

    print(result.sort_values(by=['PnL']).head())

    result.cumsum().plot()

    plt.show()


def main():
    features_list = (ft.trading_days, ft.sharpe_ratio, ft.sharpe_ratio_last_year, ft.annret, ft.annvol,
                     ft.skewness, ft.kurtosis, ft.information_ratio,
                     ft.sharpe_std, ft.sortino_ratio, ft.drawdown_area, ft.max_drawdown, ft.calmar_ratio,
                     ft.tail_ratio, ft.common_sense_ratio,
                     ft.sharpe_ratio_last_30_days, ft.sharpe_ratio_last_90_days,
                     ft.sharpe_ratio_last_150_days)

    features_csv = './data/2017-08-07-filtered-pairs-features.csv'

    pairs = pd.read_csv('./data/2017-08-03-filtered-in-sample-pairs.csv', parse_dates=True, index_col=0)

    if not os.path.exists(features_csv):
        print('File %s does not exist, creating and saving' % features_csv)

        X = create_inputs(pairs, features_list)
        y = create_outputs(pairs)

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

        observations_scaled.to_csv(features_csv)
    else:
        print('File %s exists!' % features_csv)
        observations_scaled = pd.read_csv(features_csv, index_col=0)

    # Throw out observations which are more than 4 standard deviations away from the mean
    # Note: doesn't really matter for Random Forests
    # observations_scaled = observations_scaled[np.all(observations_scaled.abs() <= 4, axis=1)]

    X = observations_scaled.drop(['OUTPUT'], axis=1)
    y = observations_scaled['OUTPUT']

    # classification_forest(X, y, features_list)
    # regression_forest(X, y, features_list)

    optimize(observations_scaled, pairs)

if __name__ == '__main__':
    main()
