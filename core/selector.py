"""
选股模块：行业Z-Score打分 + 风格分池 + 分层选股 + 行业上限
"""
import pandas as pd
from config.settings import (
    MICRO_FACTOR_WEIGHT, MICRO_FACTOR_DIRECTION,
    SELECT_STOCK_NUM, A_SHARE_MIN_COUNT, INDUSTRY_MAX_WEIGHT,
)
from core.industry_zscore import industry_zscore_standardize
from core.style_rotation import split_style_pools, allocate_seats


def compute_micro_scores(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    行业Z-Score标准化 + 加权总分

    Returns:
        DataFrame with total_score column added
    """
    factor_cols = list(MICRO_FACTOR_WEIGHT.keys())
    available_cols = [c for c in factor_cols if c in factor_df.columns]
    directions = {c: MICRO_FACTOR_DIRECTION.get(c, "positive") for c in available_cols}

    df = industry_zscore_standardize(factor_df, available_cols, directions)

    # 加权求和
    df["total_score"] = 0.0
    total_weight = sum(abs(MICRO_FACTOR_WEIGHT[c]) for c in available_cols)
    if total_weight == 0:
        total_weight = 1.0

    for col in available_cols:
        weight = abs(MICRO_FACTOR_WEIGHT[col])
        if f"{col}_norm" in df.columns:
            df["total_score"] += df[f"{col}_norm"] * weight / total_weight

    df = df.sort_values("total_score", ascending=False)
    return df


def select_target_stocks_v2(
    scored_df: pd.DataFrame,
    style_allocation: dict,
    total_count: int = SELECT_STOCK_NUM,
    a_share_min: int = A_SHARE_MIN_COUNT,
) -> list:
    """
    风格分层选股 + 行业上限约束

    Args:
        scored_df: 含 total_score, code, industry, pb, roe, profit_growth, revenue_growth, avg_amount_20
        style_allocation: {"large": pct, "small": pct, "value": pct, "growth": pct}
        total_count: 总选股数量
        a_share_min: A股最少数量

    Returns:
        list of stock codes
    """
    # 构建行业映射
    industry_map = dict(zip(scored_df["code"], scored_df["industry"].fillna("其他")))

    # 风格分池
    pools = split_style_pools(scored_df, style_allocation)

    # 分配席位
    seats = allocate_seats(total_count, style_allocation)

    # 分层选取
    selected = []
    for pool_name in ["large_value", "large_growth", "small_value", "small_growth"]:
        pool = pools.get(pool_name, pd.DataFrame())
        n_seats = seats.get(pool_name, 0)

        for _, row in pool.iterrows():
            if len([c for c in selected if _pool_of(c, pools) == pool_name]) >= n_seats:
                break
            code = row["code"]
            if code in selected:
                continue

            # 行业上限检查
            ind = industry_map.get(code, "其他")
            ind_count = sum(1 for c in selected if industry_map.get(c) == ind)
            if ind_count / total_count >= INDUSTRY_MAX_WEIGHT:
                continue  # 该行业已达上限，跳过

            selected.append(code)

    # A股数量校验
    a_count = sum(1 for c in selected if len(str(c)) == 6)
    if a_count < a_share_min:
        # 从A股池中补足
        a_pool = scored_df[scored_df["code"].str.len() == 6]
        for _, row in a_pool.iterrows():
            if a_count >= a_share_min:
                break
            code = row["code"]
            if code not in selected:
                selected.append(code)
                a_count += 1

    # 截断到目标数量
    return selected[:total_count]


def _pool_of(code, pools):
    """判断code属于哪个风格池"""
    for name, pool in pools.items():
        if code in pool["code"].values:
            return name
    return None


# ===== 兼容旧接口 =====
def select_target_stocks(factor_df, already_standardized=False):
    """
    旧接口兼容：简单加权打分选股（用于快捷测试）
    """
    from config.settings import FACTOR_WEIGHT, SELECT_STOCK_NUM, ROE_MIN, ROE_MAX
    from utils.common import drop_na_and_outlier
    from core.factor_calc import factor_standardize

    df = drop_na_and_outlier(factor_df, ROE_MIN, ROE_MAX)
    factor_cols = list(FACTOR_WEIGHT.keys())

    if not already_standardized:
        df = factor_standardize(df, factor_cols)

    df["total_score"] = 0.0
    for col, weight in FACTOR_WEIGHT.items():
        if f"{col}_norm" in df.columns:
            df["total_score"] += df[f"{col}_norm"] * weight

    df = df.sort_values("total_score", ascending=False)
    return df.head(SELECT_STOCK_NUM)["code"].tolist()
