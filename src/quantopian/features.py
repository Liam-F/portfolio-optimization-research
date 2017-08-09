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


def sharpe_n_trimester(returns, n, periods=252):
    returns = returns[-252:]
    returns_count = returns.shape[0]
    slice_size = returns_count // 4
    if n < 4:
        return sharpe_ratio(returns[(n - 1) * slice_size: n * slice_size])
    else:
        return sharpe_ratio(returns[(n - 1) * slice_size:])


def sharpe_first_trimester(returns, periods=252):
    return sharpe_n_trimester(returns, 1, periods=periods)


def sharpe_second_trimester(returns, periods=252):
    return sharpe_n_trimester(returns, 2, periods=periods)


def sharpe_third_trimester(returns, periods=252):
    return sharpe_n_trimester(returns, 3, periods=periods)


def sharpe_fourth_trimester(returns, periods=252):
    return sharpe_n_trimester(returns, 4, periods=periods)


def sharpe_ratio_last_x(returns, time_period, periods=252):
    return sharpe_ratio(returns[-time_period:], periods=periods)


def sharpe_ratio_last_90_days(returns, periods=252):
    return sharpe_ratio_last_x(returns, 90, periods=periods)


def sharpe_ratio_last_30_days(returns, periods=252):
    return sharpe_ratio_last_x(returns, 30, periods=periods)


def sharpe_ratio_last_150_days(returns, periods=252):
    return sharpe_ratio_last_x(returns, 150, periods=periods)


def sharpe_ratio_last_year(returns, periods=252):
    return sharpe_ratio_last_x(returns, 252, periods=252)


def max_rolling_sharpe(returns, window=156, periods=252):
    return returns.rolling(window).apply(sharpe_ratio).max()


def highest_6_month_sharpe(returns, periods=252):
    return max_rolling_sharpe(returns, window=156, periods=252)


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


def drawdown_pct(returns):
    if returns.empty:
        return np.nan

    c_returns = returns.cumsum()
    highs = c_returns.expanding().max()

    drawdowns = c_returns - highs

    return drawdowns[drawdowns < 0].count() / drawdowns.shape[0]


def worst_drawdown_duration(returns):
    c_returns = returns.cumsum()
    highs = c_returns.expanding().max()

    drawdowns = c_returns - highs
    worst_duration = 0
    is_underwater = False;
    current_underwater_period = 0
    for drawdown in drawdowns:
        if drawdown == 0 and is_underwater:
            is_underwater = False
        if not is_underwater and drawdown < 0:
            is_underwater = True
            current_underwater_period = 0
        if is_underwater and drawdown < 0:
            current_underwater_period += 1
        worst_duration = max(worst_duration, current_underwater_period)
    return worst_duration


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
