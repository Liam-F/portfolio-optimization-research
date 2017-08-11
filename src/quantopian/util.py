import pandas as pd
import numpy as np
import random

from matplotlib import pyplot as plt


def sample_file(filepath, sample_size):
    nb_rows = sum(1 for line in open(filepath)) - 1
    skip = sorted(random.sample(range(1, nb_rows + 1), nb_rows - sample_size))
    return pd.read_csv(filepath, parse_dates=True, index_col=0, skiprows=skip)


def drop_nan_rows(X):
    mask = np.any(np.isnan(X), axis=1)
    return X[~mask]


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
        'sharpe_ratio_first_trimester': 'Sharpe Ratio 1st trimester',
        'sharpe_ratio_second_trimester': 'Sharpe Ratio 2nd trimester',
        'sharpe_ratio_third_trimester': 'Sharpe Ratio 3rd trimester',
        'sharpe_ratio_fourth_trimester': 'Sharpe Ratio 4th trimester',
        'highest_6_month_sharpe': 'Highest 6 month rolling window Sharpe',
        'sortino_ratio': 'Sortino',
        'trading_days': 'Trading Days (PnL != 0)',
        'kurtosis': 'Kurtosis',
        'drawdown_pct': 'Percentage of drawdown periods',
        'worst_drawdown_duration': 'Worst duration of drawdown'
    })
    plt.figure()
    feature_importances.sort_values().plot(kind='barh', xerr=std, title='Feature Importance')
    plt.tight_layout()
