"""
统一回测绩效指标层（共享底座 base/backtest.py）

收敛重构 P2：把散落在 short_term/backtest_engine.py 与 short_term/backtest_runner.py 的
「胜率/盈亏比/年化/最大回撤/夏普」计算统一到这里，消除两套口径漂移。

统一口径约定：
  - 胜率 = 盈利笔数 / 总笔数（盈利 = 收益率 > 0）
  - 盈亏比 = 平均盈利 / |平均亏损|
  - 年化 = (1 + 累计收益) ^ (252 / 交易日数) - 1，交易日数 = 净值样本数 - 1
  - 最大回撤 = 净值序列 peak-to-trough 的最大跌幅（绝对值）
  - 夏普 = 日收益率均值 / 标准差 × sqrt(252)

说明：年化统一采用「净值样本数（交易日）」口径，替代 backtest_runner 原先的「自然日」口径，
使其与 backtest_engine 一致（这是本次口径统一的唯一行为变化）。
"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def win_rate(trades_pct) -> float:
    """胜率：盈利笔数占比"""
    trades_pct = np.asarray(trades_pct, dtype=float)
    if len(trades_pct) == 0:
        return 0.0
    return float((trades_pct > 0).sum() / len(trades_pct))


def profit_loss_ratio(trades_pct) -> float:
    """盈亏比 = 平均盈利 / |平均亏损|"""
    trades_pct = np.asarray(trades_pct, dtype=float)
    wins = trades_pct[trades_pct > 0]
    losses = trades_pct[trades_pct <= 0]
    if len(losses) == 0:
        return float("inf") if len(wins) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean()))
    return avg_win / avg_loss if avg_loss > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（绝对值，非负数）"""
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(abs(drawdown.min()))


def annualized_return(total_return: float, n_trading_days: int) -> float:
    """年化收益率（交易日口径）"""
    years = n_trading_days / TRADING_DAYS_PER_YEAR
    if years <= 0 or total_return <= -1:
        return float(total_return)
    return float((1 + total_return) ** (1 / years) - 1)


def sharpe_ratio(equity: pd.Series) -> float:
    """夏普比率（日收益，年化 sqrt(252)）"""
    if len(equity) < 2:
        return 0.0
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return float((daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_backtest_metrics(trades_pct, equity: pd.Series,
                             initial_capital: float, final_value: float) -> dict:
    """
    统一计算回测核心指标

    Args:
        trades_pct: 每笔交易收益率序列（list/ndarray/Series）
        equity: 净值序列（pd.Series，按日期升序，index 任意日期格式）
        initial_capital: 初始资金
        final_value: 最终净值

    Returns:
        dict: win_rate / profit_loss_ratio / total_return / annual_return /
              max_drawdown / sharpe_ratio / total_trades / avg_win_pct / avg_loss_pct
    """
    trades_pct = np.asarray(trades_pct, dtype=float)
    equity = pd.Series(equity).sort_index() if not isinstance(equity, pd.Series) else equity.sort_index()

    wins = trades_pct[trades_pct > 0]
    losses = trades_pct[trades_pct <= 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0

    total_return = (final_value - initial_capital) / initial_capital if initial_capital else 0.0
    n_days = max(len(equity) - 1, 0)

    return {
        "win_rate": win_rate(trades_pct),
        "profit_loss_ratio": profit_loss_ratio(trades_pct),
        "total_return": float(total_return),
        "annual_return": annualized_return(total_return, n_days),
        "max_drawdown": max_drawdown(equity),
        "sharpe_ratio": sharpe_ratio(equity),
        "total_trades": int(len(trades_pct)),
        "avg_win_pct": avg_win,
        "avg_loss_pct": -avg_loss,
    }
