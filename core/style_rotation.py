"""
风格轮动模块：大小盘/价值成长股票分池
"""
import numpy as np
import pandas as pd


def classify_size(market_cap_series: pd.Series) -> pd.Series:
    """
    按市值分位数分类大小盘

    Args:
        market_cap_series: 市值序列

    Returns:
        pd.Series: 'large' 或 'small'，与输入索引对齐
    """
    if len(market_cap_series) < 10:
        return pd.Series("large", index=market_cap_series.index)

    median = market_cap_series.median()
    result = market_cap_series.apply(lambda x: "large" if x >= median else "small")
    return result


def classify_value_growth(
    pb_series: pd.Series,
    roe_series: pd.Series,
    profit_growth_series: pd.Series,
    revenue_growth_series: pd.Series,
) -> pd.Series:
    """
    价值/成长分类

    价值信号: 低PB + 高ROE
    成长信号: 高利润增速 或 高收入增速

    非互斥：一只股票可能同时是价值和成长，选择信号更强的那个。

    Returns:
        pd.Series: 'value' 或 'growth'
    """
    n = len(pb_series)
    if n < 10:
        return pd.Series("value", index=pb_series.index)

    # 标准化各信号
    def zscore(s):
        m, std = s.mean(), s.std(ddof=0)
        return (s - m) / std if std > 0 else pd.Series(0, index=s.index)

    # 价值信号：低PB（取反）+ 高ROE
    pb_z = -zscore(pb_series.astype(float))
    roe_z = zscore(roe_series.astype(float))
    value_score = pb_z + roe_z

    # 成长信号：高利润增速 + 高收入增速
    pg_z = zscore(profit_growth_series.astype(float))
    rg_z = zscore(revenue_growth_series.astype(float))
    growth_score = pg_z + rg_z

    result = pd.Series("value", index=pb_series.index)
    result[growth_score > value_score] = "growth"
    return result


def split_style_pools(
    factor_df: pd.DataFrame,
    style_allocation: dict,
) -> dict:
    """
    将因子DataFrame按风格拆分为4个子池

    Args:
        factor_df: 含 total_score, pb, roe, profit_growth, revenue_growth,
                   avg_amount_20 (作为市值代理)
        style_allocation: {"large": pct, "small": pct, "value": pct, "growth": pct}

    Returns:
        {"large_value": DataFrame, "large_growth": DataFrame,
         "small_value": DataFrame, "small_growth": DataFrame}
    """
    df = factor_df.copy()

    # 大小盘分类（用成交额作为市值代理，成交额越大 ≈ 大盘）
    df["size_style"] = classify_size(df["avg_amount_20"].astype(float))

    # 价值/成长分类
    df["vg_style"] = classify_value_growth(
        df["pb"].astype(float),
        df["roe"].astype(float),
        df["profit_growth"].astype(float),
        df["revenue_growth"].astype(float),
    )

    pools = {
        "large_value": df[(df["size_style"] == "large") & (df["vg_style"] == "value")].sort_values("total_score", ascending=False),
        "large_growth": df[(df["size_style"] == "large") & (df["vg_style"] == "growth")].sort_values("total_score", ascending=False),
        "small_value": df[(df["size_style"] == "small") & (df["vg_style"] == "value")].sort_values("total_score", ascending=False),
        "small_growth": df[(df["size_style"] == "small") & (df["vg_style"] == "growth")].sort_values("total_score", ascending=False),
    }

    return pools


def allocate_seats(total_count: int, style_allocation: dict) -> dict:
    """
    根据风格比例分配各子池选股数

    Returns:
        {"large_value": N, "large_growth": N, "small_value": N, "small_growth": N}
    """
    large_pct = style_allocation.get("large", 0.5)
    small_pct = style_allocation.get("small", 0.5)
    value_pct = style_allocation.get("value", 0.5)
    growth_pct = style_allocation.get("growth", 0.5)

    lv = int(total_count * large_pct * value_pct)
    lg = int(total_count * large_pct * growth_pct)
    sv = int(total_count * small_pct * value_pct)
    sg = int(total_count * small_pct * growth_pct)

    # 处理取整误差
    allocated = {"large_value": lv, "large_growth": lg, "small_value": sv, "small_growth": sg}
    diff = total_count - sum(allocated.values())
    # 将差额加到最大的池
    max_pool = max(allocated, key=allocated.get)
    allocated[max_pool] += diff

    return allocated
