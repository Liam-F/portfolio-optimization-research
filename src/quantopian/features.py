import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_spy(fn='./data/es-all.csv'):
    data = pd.read_csv(fn, parse_dates=True, index_col=0)[['Adj Close']].rename(columns={'Adj Close': 'spy'})

    data['spy_returns'] = data['spy'].pct_change()

    return data[['spy_returns']]


def trading_days(returns):
    return returns[returns != 0].shape[0]


np.seterr(all='raise')

def sharpe_ratio(returns, periods=252):
    std = returns.std()

    if std == 0:
        return np.nan

    return np.sqrt(periods) * returns.mean() / std

def sharpe_ratio_last_year(returns, periods=252):
    return sharpe_ratio(returns.iloc[-252:], periods=periods)


def annret(returns, periods=252):
    return periods * returns.mean()


def annret_last_year(returns, periods=252):
    return annret(returns.iloc[-252:], periods=periods)


def annvol(returns, periods=252):
    return np.sqrt(periods) * returns.std()


def annvol_last_year(returns, periods=252):
    return annvol(returns.iloc[-252:], periods=periods)


def skewness(returns):
    return returns.skew()


def skewness_last_year(returns):
    return skewness(returns.iloc[-252:])


def kurtosis(returns):
    return returns.kurt()


def kurtosis_last_year(returns):
    return kurtosis(returns.iloc[-252:])


def stability(returns):
    clog_returns = np.log(returns).cumsum()

    lm = sm.OLS(clog_returns, sm.add_constant(np.arange(returns.shape[0]))).fit()

    return lm.rsquared


def beta_spy(returns):
    raise NotImplementedError


def alpha_spy(returns):
    return NotImplementedError


def information_ratio(returns):
    spy_returns = load_spy()

    data = returns.rename('returns').to_frame().merge(spy_returns, left_index=True, right_index=True)
    diff = data['returns'] - data['spy_returns']

    return diff.mean() / diff.std()


def information_ratio_last_year(returns):
    return information_ratio(returns.iloc[-252:])


def beta_std(returns):
    raise NotImplementedError


def sharpe_std(returns, periods=252, window=126):
    roll = returns.rolling(window=window)

    roll_sharpe = (np.sqrt(periods) * roll.mean() / roll.std()).dropna()

    return roll_sharpe.std()


def sortino_ratio(returns):
    return returns.mean() / returns[returns < 0].std()


def sortino_last_year(returns):
    return sortino_ratio(returns.iloc[-252:])


def drawdown_area(returns):
    c_returns = returns.cumsum()
    highs = c_returns.expanding().max()

    diff = c_returns - highs

    return diff.abs().sum()


def drawdown_area_last_year(returns):
    return drawdown_area(returns.iloc[-252:])


def max_drawdown(returns):
    c_returns = returns.cumsum()
    highs = c_returns.expanding().max()

    diff = c_returns - highs

    return np.abs(np.min(diff))


def max_drawdown_last_year(returns):
    return max_drawdown(returns.iloc[-252:])


def calmar_ratio(returns, periods=252):
    md = max_drawdown(returns)

    if md == 0:
        return np.nan

    return periods * returns.mean() / md


def calmar_ratio_last_year(returns, periods=252):
    return calmar_ratio(returns.iloc[-252:], periods=periods)


def tail_ratio(returns, tail=0.05):
    qt = np.abs(returns.quantile(0.05))

    if qt == 0:
        return np.nan

    return returns.quantile(1 - tail) / qt


def tail_ratio_last_year(returns, tail=0.05):
    return tail_ratio(returns.iloc[-252:], tail=tail)


def common_sense_ratio(returns, tail=0.05, periods=252):
    return tail_ratio(returns, tail=tail) * (1 + annret(returns, periods))


def common_sense_ratio_last_year(returns, tail=0.05, periods=252):
    return common_sense_ratio(returns.iloc[-252:], tail=tail, periods=periods)

