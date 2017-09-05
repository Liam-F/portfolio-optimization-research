import numpy as np
import pandas as pd


def load_spy(fn='./data/instruments/es-all.csv'):
    data = pd.read_csv(fn, parse_dates=True, index_col=0)[['Adj Close']].rename(columns={'Adj Close': 'spy'})

    data['spy_returns'] = data['spy'].pct_change()

    return data[['spy_returns']]


def trading_days(pnls):
    return pnls[pnls != 0].shape[0]


np.seterr(all='raise')


def sharpe_ratio(pnls, periods=252):
    std = pnls.std()

    if std == 0:
        return np.nan

    return np.sqrt(periods) * pnls.mean() / std


def sharpe_n_trimester(pnls, n, periods=252):
    pnls = pnls[-252:]
    returns_count = pnls.shape[0]
    slice_size = returns_count // 4
    if n < 4:
        return sharpe_ratio(pnls[(n - 1) * slice_size: n * slice_size])
    else:
        return sharpe_ratio(pnls[(n - 1) * slice_size:])


def sharpe_first_trimester(pnls, periods=252):
    return sharpe_n_trimester(pnls, 1, periods=periods)


def sharpe_second_trimester(pnls, periods=252):
    return sharpe_n_trimester(pnls, 2, periods=periods)


def sharpe_third_trimester(pnls, periods=252):
    return sharpe_n_trimester(pnls, 3, periods=periods)


def sharpe_fourth_trimester(pnls, periods=252):
    return sharpe_n_trimester(pnls, 4, periods=periods)


def sharpe_ratio_last_x(pnls, time_period, periods=252):
    return sharpe_ratio(pnls[-time_period:], periods=periods)


def sharpe_ratio_last_90_days(pnls, periods=252):
    return sharpe_ratio_last_x(pnls, 90, periods=periods)


def sharpe_ratio_last_30_days(pnls, periods=252):
    return sharpe_ratio_last_x(pnls, 30, periods=periods)


def sharpe_ratio_last_150_days(pnls, periods=252):
    return sharpe_ratio_last_x(pnls, 150, periods=periods)


def sharpe_ratio_last_year(pnls, periods=252):
    return sharpe_ratio_last_x(pnls, 252, periods=252)


def max_rolling_sharpe(pnls, window=156, periods=252):
    return pnls.rolling(window).apply(sharpe_ratio).max()


def highest_6_month_sharpe(pnls, periods=252):
    return max_rolling_sharpe(pnls, window=156, periods=252)


def annret(pnls, periods=252):
    return periods * pnls.mean()


def annret_last_year(pnls, periods=252):
    return annret(pnls.iloc[-252:], periods=periods)


def annvol(pnls, periods=252):
    return np.sqrt(periods) * pnls.std()


def annvol_last_year(pnls, periods=252):
    return annvol(pnls.iloc[-252:], periods=periods)


def skewness(pnls):
    return pnls.skew()


def skewness_last_year(pnls):
    return skewness(pnls.iloc[-252:])


def kurtosis(pnls):
    return pnls.kurt()


def kurtosis_last_year(pnls):
    return kurtosis(pnls.iloc[-252:])


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


def beta_std(pnls):
    raise NotImplementedError


def sharpe_std(pnls, periods=252, window=126):
    roll = pnls.rolling(window=window)

    roll_sharpe = (np.sqrt(periods) * roll.mean() / roll.std()).dropna()

    return roll_sharpe.std()


def sortino_ratio(pnls):
    return pnls.mean() / pnls[pnls < 0].std()


def sortino_last_year(pnls):
    return sortino_ratio(pnls.iloc[-252:])


def drawdown_pct(pnls):
    if pnls.empty:
        return np.nan

    c_returns = pnls.cumsum()
    highs = c_returns.expanding().max()

    drawdowns = c_returns - highs

    return drawdowns[drawdowns < 0].count() / drawdowns.shape[0]


def worst_drawdown_duration(pnls):
    c_returns = pnls.cumsum()
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


def drawdown_area(pnls):
    c_returns = pnls.cumsum()
    highs = c_returns.expanding().max()

    diff = c_returns - highs

    return diff.abs().sum()


def drawdown_area_last_year(pnls):
    return drawdown_area(pnls.iloc[-252:])


def max_drawdown(pnls):
    c_returns = pnls.cumsum()
    highs = c_returns.expanding().max()

    diff = c_returns - highs

    return np.abs(np.min(diff))


def max_drawdown_last_year(pnls):
    return max_drawdown(pnls.iloc[-252:])


def calmar_ratio(pnls, periods=252):
    md = max_drawdown(pnls)

    if md == 0:
        return np.nan

    return periods * pnls.mean() / md


def calmar_ratio_last_year(pnls, periods=252):
    return calmar_ratio(pnls.iloc[-252:], periods=periods)


def tail_ratio(pnls, tail=0.05):
    qt = np.abs(pnls.quantile(0.05))

    if qt == 0:
        return np.nan

    return pnls.quantile(1 - tail) / qt


def tail_ratio_last_year(pnls, tail=0.05):
    return tail_ratio(pnls.iloc[-252:], tail=tail)


def common_sense_ratio(pnls, tail=0.05, periods=252):
    return tail_ratio(pnls, tail=tail) * (1 + annret(pnls, periods))


def common_sense_ratio_last_year(pnls, tail=0.05, periods=252):
    return common_sense_ratio(pnls.iloc[-252:], tail=tail, periods=periods)
