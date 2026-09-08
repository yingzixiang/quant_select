import pandas as pd
import numpy as np


def drop_na_and_outlier(df, roe_min, roe_max):
    """空值+异常值过滤"""
    df = df.dropna()
    df = df[(df["pb"] > 0) & (df["roe"] > roe_min) & (df["roe"] < roe_max)]
    return df


def get_trade_date_info(dt):
    """格式化日期输出"""
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y-%m-%d")
    return str(dt)[:10]


def safe_float(val, default=None):
    """安全转换为float，失败返回default"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def percentile_rank(value, history):
    """计算value在history中的分位数 [0, 1]"""
    if history is None or len(history) == 0:
        return 0.5
    arr = np.sort(np.asarray(history, dtype=float))
    return float(np.searchsorted(arr, value) / len(arr))


def annualize_volatility(daily_returns, trading_days=252):
    """年化波动率"""
    return float(np.std(daily_returns, ddof=0) * np.sqrt(trading_days))


def compute_yoy(current, previous):
    """计算同比增长率"""
    if previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous)
