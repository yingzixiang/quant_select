"""
微观因子模块：12项因子 → 两轮计算

第一轮（批量，全市场）:
  stock_yjbb_em(date) → ROE, 利润增速, 收入增速, 毛利率, 每股收益, 行业
  → 6项因子 + PE代理 → 加权打分 → Top 200

第二轮（逐只，仅Top200）:
  stock_zh_valuation_baidu → PB
  stock_zh_a_hist → 动量/波动率/成交额
  stock_history_dividend_detail → 股息率
"""
import sys
import datetime
import numpy as np
import pandas as pd
import akshare as ak

from config.settings import (
    MICRO_FACTOR_WEIGHT, ROE_MIN, ROE_MAX, MIN_DAILY_AMOUNT,
    PRICE_CACHE_TTL,
)
from core.data_cache import cached_api_call


def _batch_financial_to_dict(date: str) -> dict:
    """
    批量取财务数据 → dict {code: {factor: value}}
    包含: roe, profit_growth, revenue_growth, gross_margin, eps, industry
    """
    print(f"  [微观] 批量拉取财务数据 (stock_yjbb_em, date={date})...")
    try:
        df = ak.stock_yjbb_em(date=date)
    except Exception as e:
        print(f"  [微观] 财务数据拉取失败: {e}")
        return {}

    result = {}
    for _, row in df.iterrows():
        code = str(row["股票代码"]).zfill(6)
        try:
            result[code] = {
                "roe": safe_float(row.get("净资产收益率")),
                "profit_growth": safe_float(row.get("净利润-同比增长")),
                "revenue_growth": safe_float(row.get("营业总收入-同比增长")),
                "gross_margin": safe_float(row.get("销售毛利率")),
                "eps": safe_float(row.get("每股收益")),
                "industry": str(row.get("所处行业", "")),
            }
        except Exception:
            continue
    print(f"  [微观] 财务数据覆盖 {len(result)} 只股票")
    sys.stdout.flush()
    return result


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _get_price_factors(code: str, start_date: str, end_date: str) -> dict:
    """获取单只股票的量价因子：动量/波动率/成交额（使用Tencent数据源）"""
    try:
        # 转换symbol格式: 000001 → sz000001 或 sh600000
        if code.startswith(("6", "9")):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"

        df = cached_api_call(
            ak.stock_zh_a_hist_tx,
            symbol=symbol, start_date=start_date, end_date=end_date,
            max_age_seconds=PRICE_CACHE_TTL,
        )
        if df is None or len(df) < 60:
            return None

        close = df["close"].astype(float)
        # amount 是成交量(手), 成交额 = 成交量 * 收盘价 * 100
        amount_value = df["amount"].astype(float) * close * 100

        # 60日动量
        mom60 = float(close.pct_change(59).iloc[-1]) if len(close) >= 60 else None

        # 30日波动率（年化）
        if len(close) >= 30:
            daily_ret = close.pct_change().dropna().tail(30)
            vol30 = float(np.std(daily_ret, ddof=0) * np.sqrt(252))
        else:
            vol30 = None

        # 20日均成交额
        if len(amount_value) >= 20:
            avg_amt20 = float(amount_value.tail(20).mean())
        else:
            avg_amt20 = None

        # 最新收盘价（用于PE/股息率计算）
        latest_price = float(close.iloc[-1])

        return {
            "mom_60": mom60,
            "volatility_30": vol30,
            "avg_amount_20": avg_amt20,
            "latest_price": latest_price,
        }
    except Exception:
        return None


def _get_pb(code: str) -> float:
    """获取单只股票PB"""
    try:
        df = cached_api_call(
            ak.stock_zh_valuation_baidu,
            symbol=code, indicator="市净率", period="近一年",
            max_age_seconds=86400,
        )
        if df is not None and not df.empty:
            return safe_float(df["value"].iloc[-1])
        return None
    except Exception:
        return None


def _get_dividend_yield(code: str, latest_price: float) -> float:
    """计算TTM股息率 = 近12个月派息 / 当前股价"""
    try:
        df = cached_api_call(
            ak.stock_history_dividend_detail,
            symbol=code,
            max_age_seconds=86400 * 7,
        )
        if df is None or df.empty:
            return None
        # 筛选近12个月的派息
        if "派息" not in df.columns:
            return None
        # 取最近一次派息记录
        latest = df[df["派息"].notna()].head(1)
        if latest.empty:
            return None
        dividend_per_share = safe_float(latest["派息"].iloc[0])
        if dividend_per_share is None or latest_price is None or latest_price <= 0:
            return None
        # 简单估算：最近一次派息 / 当前股价
        return dividend_per_share / latest_price
    except Exception:
        return None


def compute_all_micro_factors(
    stock_codes: list,
    financial_date: str = "20260331",
    price_start: str = "20260201",
    price_end: str = "20260526",
    top_n: int = 200,
) -> pd.DataFrame:
    """
    两轮计算全市场12项微观因子

    Returns:
        DataFrame: code, industry, pb, pe_ttm, dividend_yield, roe, debt_ratio,
                   gross_margin_yoy, profit_growth, revenue_growth,
                   rd_intensity_growth, mom_60, volatility_30, avg_amount_20
    """
    total_codes = len(stock_codes)
    print(f"\n  [微观] ===== 12因子计算开始（{total_codes} 只标的）=====")

    # ===== 第一轮：批量财务（6项因子）=====
    fin_dict = _batch_financial_to_dict(financial_date)

    # 构建第一轮DataFrame
    round1_data = []
    for code in stock_codes:
        fin = fin_dict.get(code)
        if fin is None:
            continue
        roe = fin["roe"]
        profit_growth = fin["profit_growth"]
        revenue_growth = fin["revenue_growth"]
        gross_margin = fin["gross_margin"]
        eps = fin["eps"]
        industry = fin["industry"]

        if roe is None or roe < ROE_MIN or roe > ROE_MAX:
            continue

        round1_data.append({
            "code": code,
            "industry": industry,
            "roe": roe,
            "profit_growth": profit_growth or 0,
            "revenue_growth": revenue_growth or 0,
            "gross_margin": gross_margin or 0,
            "eps": eps or 0,
            # 以下占位，第二轮补全
            "pb": None,
            "pe_ttm": None,
            "dividend_yield": None,
            "debt_ratio": 0,       # 默认中性（无批量API）
            "gross_margin_yoy": 0, # 暂无同比数据，用毛利率水平值代理
            "rd_intensity_growth": 0,  # 无API，中性
            "mom_60": None,
            "volatility_30": None,
            "avg_amount_20": None,
        })

    if not round1_data:
        print("  [微观] 第一轮：无有效数据！")
        return pd.DataFrame()

    df1 = pd.DataFrame(round1_data)
    print(f"  [微观] 第一轮（批量财务）：{len(df1)} 只有效标的")

    # 第一轮打分（仅用已有因子，等权）
    round1_cols = ["roe", "profit_growth", "revenue_growth", "gross_margin"]
    for col in round1_cols:
        series = df1[col].astype(float)
        mean = series.mean()
        std = series.std(ddof=0)
        if std > 0:
            df1[f"{col}_z"] = (series - mean) / std
        else:
            df1[f"{col}_z"] = 0

    df1["round1_score"] = (
        df1["roe_z"] * 0.30 +
        df1["profit_growth_z"] * 0.30 +
        df1["revenue_growth_z"] * 0.20 +
        df1["gross_margin_z"] * 0.20
    )
    df1 = df1.sort_values("round1_score", ascending=False)

    # 取Top N进入第二轮
    top_codes = df1.head(top_n)["code"].tolist()
    print(f"  [微观] 第二轮（逐只PB+量价）：Top {len(top_codes)} 只")

    # ===== 第二轮：逐只补全 =====
    pb_map = {}
    price_map = {}
    div_map = {}

    # 逐只拉取PB + 价格（串行，避免Sina rate-limit）
    completed = 0
    for code in top_codes:
        try:
            pb = _get_pb(code)
            if pb is not None and pb > 0:
                pb_map[code] = pb

            pf = _get_price_factors(code, price_start, price_end)
            if pf is not None:
                price_map[code] = pf

            completed += 1
            if completed % 50 == 0 or completed == len(top_codes):
                print(f"  [微观] 第二轮进度: {completed}/{len(top_codes)}"
                      f"（PB有效:{len(pb_map)} 量价有效:{len(price_map)}）")
                sys.stdout.flush()
        except Exception:
            completed += 1

    # 股息率（对已有价格的标的拉取，串行避免频率过高）
    for code in top_codes:
        if code in price_map:
            div_yield = _get_dividend_yield(code, price_map[code]["latest_price"])
            if div_yield is not None:
                div_map[code] = div_yield

    print(f"  [微观] 第二轮完成：PB覆盖{len(pb_map)} 量价覆盖{len(price_map)} 股息覆盖{len(div_map)}")

    # ===== 合并结果 =====
    for idx in df1.index:
        code = df1.loc[idx, "code"]

        if code in pb_map:
            df1.loc[idx, "pb"] = pb_map[code]

        if code in price_map:
            pf = price_map[code]
            df1.loc[idx, "mom_60"] = pf["mom_60"]
            df1.loc[idx, "volatility_30"] = pf["volatility_30"]
            df1.loc[idx, "avg_amount_20"] = pf["avg_amount_20"]

            # PE-TTM代理：当前股价 / (每股收益 * 4)
            if pf.get("latest_price") and df1.loc[idx, "eps"] > 0.01:
                df1.loc[idx, "pe_ttm"] = pf["latest_price"] / (df1.loc[idx, "eps"] * 4)

        if code in div_map:
            df1.loc[idx, "dividend_yield"] = div_map[code]

    # 毛利率同比：暂用毛利率水平值作为代理（后续可用上季度数据计算真实同比）
    df1["gross_margin_yoy"] = df1["gross_margin"].astype(float)

    # 过滤：必须有PB和量价因子
    df_final = df1.dropna(subset=["pb", "mom_60", "volatility_30", "avg_amount_20"])
    df_final = df_final[df_final["pb"] > 0]
    df_final = df_final[df_final["avg_amount_20"] >= MIN_DAILY_AMOUNT]

    # 异常值过滤
    df_final = df_final[df_final["pe_ttm"].isna() | ((df_final["pe_ttm"] > 0) & (df_final["pe_ttm"] <= 100))]

    # 填充剩余缺失为中位数
    for col in ["dividend_yield", "pe_ttm"]:
        median_val = df_final[col].median()
        df_final[col] = df_final[col].fillna(median_val if not pd.isna(median_val) else 0)

    df_final = df_final.reset_index(drop=True)
    print(f"  [微观] 最终有效因子数据：{len(df_final)} 只")
    sys.stdout.flush()

    return df_final
