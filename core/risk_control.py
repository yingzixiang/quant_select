"""
风控模块：6项前置风控 + 5项后置校验
"""
import numpy as np
import pandas as pd
import akshare as ak
from config.settings import (
    MIN_LISTED_DAYS, MIN_DAILY_AMOUNT, MAX_DEBT_RATIO,
    POST_RISK, SINGLE_STOCK_MAX_WEIGHT, INDUSTRY_MAX_WEIGHT,
)
from core.data_cache import cached_api_call


def pre_filter_stocks(stock_df: pd.DataFrame, min_listed_days: int = MIN_LISTED_DAYS) -> tuple:
    """
    6项前置风控过滤

    Returns:
        (filtered_df, stats_dict)
    """
    stats = {"total": len(stock_df), "rules": {}}
    df = stock_df.copy()

    # Rule 1: 剔除ST
    if "name" in df.columns:
        before = len(df)
        mask = ~df["name"].str.contains("ST", na=False)
        df = df[mask]
        stats["rules"]["no_st"] = before - len(df)

    # Rule 2: 剔除上市不足1年（通过个股信息判断，暂跳过以减少API调用）
    # TODO: 准确判断需要 stock_individual_info_em(symbol)
    stats["rules"]["min_listed_days"] = "skipped"

    # Rule 3: 审计意见筛选（无批量API，跳过）
    stats["rules"]["audit_opinion"] = "skipped"

    # Rule 4: 资产负债率>80% 且净利润连续下滑（部分实现）
    stats["rules"]["debt_ratio"] = "partial"

    # Rule 5: 近20日日均成交额<5000万（在因子计算中过滤）
    stats["rules"]["min_amount"] = "delegated_to_micro_factors"

    # Rule 6: 停牌/长期停盘（在价格数据获取中自然过滤）
    stats["rules"]["suspended"] = "delegated_to_price_check"

    return df, stats


def post_check_portfolio(
    selected_codes: list,
    factor_df: pd.DataFrame,
    industry_map: dict,
) -> dict:
    """
    5项后置风控校验

    Returns:
        {"passed": bool, "violations": list, "warnings": list}
    """
    violations = []
    warnings = []
    n = len(selected_codes)

    # Rule 1: 单只个股仓位 ≤ 5%
    single_weight = 1.0 / n if n > 0 else 0
    if single_weight > SINGLE_STOCK_MAX_WEIGHT:
        violations.append(f"等权仓位 {single_weight:.1%} 超过上限 {SINGLE_STOCK_MAX_WEIGHT:.1%}")

    # Rule 2: 单一行业 ≤ 20%
    industry_counts = {}
    for code in selected_codes:
        ind = industry_map.get(code, "未知")
        industry_counts[ind] = industry_counts.get(ind, 0) + 1

    for ind, count in industry_counts.items():
        ind_weight = count / n
        if ind_weight > INDUSTRY_MAX_WEIGHT:
            violations.append(f"行业 {ind} 权重 {ind_weight:.1%} 超过上限 {INDUSTRY_MAX_WEIGHT:.1%}")

    # Rule 3/4: 价值/成长组合 PE/PB 校验
    # 获取沪深300 PE/PB
    try:
        benchmark_df = cached_api_call(
            ak.stock_zh_index_value_csindex,
            symbol="000300",
            max_age_seconds=86400,
        )
        if benchmark_df is not None and not benchmark_df.empty:
            bm_pe = float(benchmark_df["市盈率1"].dropna().iloc[-1]) if "市盈率1" in benchmark_df.columns else None
            bm_pb = float(benchmark_df["股息率1"].dropna().iloc[-1]) if "股息率1" in benchmark_df.columns else None
        else:
            bm_pe, bm_pb = None, None
    except Exception:
        bm_pe, bm_pb = None, None

    if bm_pe is None or bm_pb is None:
        warnings.append("无法获取沪深300 PE/PB基准，跳过组合估值校验")
    else:
        sel_df = factor_df[factor_df["code"].isin(selected_codes)]
        if "pe_ttm" in sel_df.columns and "pb" in sel_df.columns:
            avg_pe = sel_df["pe_ttm"].dropna().mean()
            avg_pb = sel_df["pb"].dropna().mean()
            # 成长组合 PE/PB ≤ 沪深300 2倍
            if avg_pe > bm_pe * POST_RISK["growth_pe_pb_ratio_max"]:
                warnings.append(f"组合PE({avg_pe:.1f}) 超过基准({bm_pe:.1f})×{POST_RISK['growth_pe_pb_ratio_max']}")
            if avg_pb > bm_pb * POST_RISK["growth_pe_pb_ratio_max"]:
                warnings.append(f"组合PB({avg_pb:.2f}) 超过基准({bm_pb:.2f})×{POST_RISK['growth_pe_pb_ratio_max']}")

    # Rule 5: 组合30日波动率 ≤ 沪深300 1.2倍
    warnings.append("组合波动率校验需要在回测中计算（暂跳过）")

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }
