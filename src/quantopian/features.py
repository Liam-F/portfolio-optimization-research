import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_spy(fn='./data/es-all.csv'):
    data = pd.read_csv(fn, parse_dates=True, index_col=0)[['Adj Close']].rename(columns={'Adj Close': 'spy'})

    data['spy_returns'] = data['spy'].pct_change()

    return data[['spy_returns']]



def trading_days(returns):
    return returns.shape[0]


def sharpe_ratio(returns, periods=252):
    return np.sqrt(periods) * returns.mean() / returns.std()


def sharpe_ratio_last_year(returns, periods=252):
    return sharpe_ratio(returns.iloc[-252:], periods=periods)


def annret(returns, periods=252):
    return periods * returns.mean()


def annvol(returns, periods=252):
    return np.sqrt(periods) * returns.std()


def skewnewss(returns):
    return returns.skew()


def kurtosis(returns):
    return returns.kurt()


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


def beta_std(returns):
    raise NotImplementedError


def sharpe_std(returns, periods=252, window=126):
    roll = returns.rolling(window=window)

    roll_sharpe = (np.sqrt(periods) * roll.mean() / roll.std()).dropna()

    return roll_sharpe.std()


def sortino_ratio(returns):
    return returns.mean() / returns[returns < 0].std()


def drawdown_area(returns):
    c_returns = returns.cumsum()
    highs = c_returns.expanding().max()

    diff = c_returns - highs

    return diff.abs().sum()


def max_drawdown(returns):
    c_returns = returns.cumsum()
    highs = c_returns.expanding().max()

    diff = c_returns - highs

    return np.abs(np.min(diff))


def calmar_ratio(returns, periods=252):
    return periods * returns.mean() / max_drawdown(returns)


def tail_ratio(returns, tail=0.05):
    return returns.quantile(1 - tail) / np.abs(returns.quantile(0.05))


def common_sense_ratio(returns, tail=0.05, periods=252):
    return tail_ratio(returns, tail=tail) * (1 + annret(returns, periods))

