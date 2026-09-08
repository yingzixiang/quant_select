"""
自定义短线因子：量价结构 + 资金流向 + 日内强弱 + RPS
基于《短线量化选股》指南的高阶因子库
"""
import numpy as np
import pandas as pd
import sys


def safe_div(a, b, fallback=0.0):
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return fallback
    return a / b


# ============================================================
# 量价结构因子
# ============================================================

def factor_vol_price_corr_diff(kline_df):
    """
    量价相关性差：corr(open,vol,5) - corr(open,vol,20)
    短期量价共振 vs 中期量价背离
    """
    if kline_df is None or len(kline_df) < 20:
        return 0.0

    o = kline_df["open"].astype(float)
    v = kline_df["volume"].astype(float)

    corr5 = o.tail(5).corr(v.tail(5))
    corr20 = o.tail(20).corr(v.tail(20))

    diff = corr5 - corr20
    return diff if not np.isnan(diff) else 0.0


def factor_vol_price_slope(kline_df):
    """
    量价斜率：vol_change / close_change
    衡量量价配合效率
    """
    if kline_df is None or len(kline_df) < 2:
        return 0.0

    vol = kline_df["volume"].astype(float)
    close = kline_df["close"].astype(float)

    vol_chg = vol.iloc[-1] - vol.iloc[-2]
    price_chg = close.iloc[-1] - close.iloc[-2]

    if abs(price_chg) < 1e-6:
        return 0.0 if abs(vol_chg) < 1e-6 else (1.0 if vol_chg > 0 else -1.0)

    return vol_chg / price_chg


def factor_breakout_volume(kline_df):
    """
    放量突破信号：今日收盘 > 20日最高 且 成交量 > 5日均量*1.3
    返回突破强度
    """
    if kline_df is None or len(kline_df) < 21:
        return 0.0

    close = kline_df["close"].astype(float)
    vol = kline_df["volume"].astype(float)

    is_breakout = close.iloc[-1] > close.iloc[-21:-1].max()
    vol_ratio = vol.iloc[-1] / vol.tail(6).head(5).mean() if vol.tail(6).head(5).mean() > 0 else 0

    if is_breakout and vol_ratio > 1.3:
        return min(vol_ratio / 2.0, 1.0)  # 归一化到0~1
    return 0.0


def factor_shrink_pullback(kline_df):
    """
    缩量回调信号：(收盘<昨日收盘) 且 (成交量<5日均量*0.7)
    返回回调健康度（缩量越明显越好）
    """
    if kline_df is None or len(kline_df) < 6:
        return 0.0

    close = kline_df["close"].astype(float)
    vol = kline_df["volume"].astype(float)

    is_pullback = close.iloc[-1] < close.iloc[-2]
    avg_vol_5 = vol.tail(5).mean()
    vol_ratio = vol.iloc[-1] / avg_vol_5 if avg_vol_5 > 0 else 1.0

    if is_pullback and vol_ratio < 0.7:
        return (0.7 - vol_ratio) / 0.7  # 缩量越明显，得分越高
    return 0.0


def factor_volume_breakout_strength(kline_df):
    """
    放量强度：今日成交量 / max(前5日均量, 前20日均量)
    """
    if kline_df is None or len(kline_df) < 20:
        return 0.0

    vol = kline_df["volume"].astype(float)
    ma5 = vol.tail(6).head(5).mean()
    ma20 = vol.tail(21).head(20).mean()
    baseline = max(ma5, ma20)
    if baseline <= 0:
        return 0.0

    ratio = vol.iloc[-1] / baseline
    return min(ratio / 3.0, 1.0)  # 归一化，3倍以上封顶


# ============================================================
# 日内强弱因子
# ============================================================

def factor_candle_body_ratio(kline_df):
    """阳线实体占比：(close-open)/(high-low)"""
    open_ = kline_df["open"].astype(float).iloc[-1]
    high = kline_df["high"].astype(float).iloc[-1]
    low = kline_df["low"].astype(float).iloc[-1]
    close = kline_df["close"].astype(float).iloc[-1]

    denom = high - low
    if abs(denom) < 1e-6:
        return 0.0
    return (close - open_) / denom


def factor_upper_shadow(kline_df):
    """上影线占比"""
    open_ = kline_df["open"].astype(float).iloc[-1]
    high = kline_df["high"].astype(float).iloc[-1]
    low = kline_df["low"].astype(float).iloc[-1]
    close = kline_df["close"].astype(float).iloc[-1]

    denom = high - low
    if abs(denom) < 1e-6:
        return 0.0
    body_high = max(open_, close)
    return (high - body_high) / denom


def factor_lower_shadow(kline_df):
    """下影线占比"""
    open_ = kline_df["open"].astype(float).iloc[-1]
    high = kline_df["high"].astype(float).iloc[-1]
    low = kline_df["low"].astype(float).iloc[-1]
    close = kline_df["close"].astype(float).iloc[-1]

    denom = high - low
    if abs(denom) < 1e-6:
        return 0.0
    body_low = min(open_, close)
    return (body_low - low) / denom


def factor_open_gap(kline_df):
    """今日开盘相对昨日收盘的跳空幅度"""
    if len(kline_df) < 2:
        return 0.0
    open_ = kline_df["open"].astype(float).iloc[-1]
    prev_close = kline_df["close"].astype(float).iloc[-2]
    if prev_close == 0:
        return 0.0
    return open_ / prev_close - 1


def factor_intraday_volatility(kline_df):
    """日内振幅: (high-low)/close"""
    high = kline_df["high"].astype(float).iloc[-1]
    low = kline_df["low"].astype(float).iloc[-1]
    close = kline_df["close"].astype(float).iloc[-1]
    if close == 0:
        return 0.0
    return (high - low) / close


# ============================================================
# 资金流向因子（从money_flow_dict获取）
# ============================================================

def compute_money_flow_factors(money_flow_dict: dict, kline_dict: dict) -> pd.DataFrame:
    """
    计算资金流向因子

    Args:
        money_flow_dict: {code: {main_net_inflow, big_net_inflow, ...}}
        kline_dict: {code: kline_df} 用于计算成交额

    Returns:
        DataFrame: index=code
    """
    print("[自定义因子] 计算资金流向因子...")
    rows = []

    for code, mf in money_flow_dict.items():
        if code not in kline_dict:
            continue

        kdf = kline_dict[code]
        close = kdf["close"].astype(float)
        vol = kdf["volume"].astype(float)

        # 当日成交额
        day_amount = close.iloc[-1] * vol.iloc[-1] if len(close) > 0 else 1

        # 5日均成交额
        if len(close) >= 5:
            avg_amount_5 = (close.tail(5) * vol.tail(5)).mean()
        else:
            avg_amount_5 = day_amount

        main_inflow = mf.get("main_net_inflow", 0)
        big_net = mf.get("big_net_inflow", 0)

        # 主力净流入率
        main_inflow_rate = safe_div(main_inflow, day_amount)

        # 大单买入强度
        big_buy = mf.get("big_buy", 0)
        big_sell = mf.get("big_sell", 0)
        big_buy_strength = safe_div(big_buy, big_buy + big_sell) if (big_buy + big_sell) > 0 else 0.5

        # 主力净流入占成交额比（标准化）
        main_ratio = safe_div(main_inflow, avg_amount_5)

        rows.append({
            "code": code,
            "main_inflow_rate": main_inflow_rate,
            "big_buy_strength": big_buy_strength,
            "main_net_inflow_ratio": main_ratio,
            "main_net_inflow_raw": main_inflow,
            "big_net_inflow_raw": big_net,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("code")
    print(f"[自定义因子] 资金流向因子完成，覆盖 {len(result)} 只")
    sys.stdout.flush()
    return result


# ============================================================
# 资金趋势因子（需要多日资金数据，此处做简化近似）
# ============================================================

def compute_money_trend_factors(kline_dict: dict) -> pd.DataFrame:
    """
    基于量价数据近似计算3日资金趋势
    用 3日涨跌幅 * 3日量比 作为主力资金方向的代理变量

    真实环境应使用逐日资金流向数据计算: sum(net_inflow[-3:]) / sum(abs(net_inflow[-3:]))
    """
    print("[自定义因子] 计算资金趋势因子（量价代理）...")
    rows = []

    for code, df in kline_dict.items():
        if df is None or len(df) < 5:
            continue

        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        ret3 = close.pct_change(3).iloc[-1] if len(close) >= 4 else 0
        vol_ratio3 = vol.tail(3).mean() / vol.tail(6).head(3).mean() if len(vol) >= 6 else 1

        rows.append({
            "code": code,
            "money_trend_3d": ret3 * vol_ratio3,
            "ret_3d": ret3,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("code")
    print(f"[自定义因子] 资金趋势完成，覆盖 {len(result)} 只")
    sys.stdout.flush()
    return result


# ============================================================
# RPS因子（需要指数数据）
# ============================================================

def compute_rps_factors(kline_dict: dict, index_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算个股RPS（相对强弱指标）

    RPS(20) = 个股20日涨幅 在 全市场20日涨幅 中的百分位排名 * 100

    Args:
        kline_dict: {code: kline_df}
        index_df: 指数日线 DataFrame（用于计算个股相对指数的超额收益）

    Returns:
        DataFrame: index=code, columns=[rps_20, rps_50, excess_return_20d]
    """
    print("[自定义因子] 计算RPS因子...")

    rows = []
    for code, df in kline_dict.items():
        if df is None or len(df) < 50:
            continue

        close = df["close"].astype(float)

        ret20 = close.pct_change(20).iloc[-1] if len(close) >= 21 else 0
        ret50 = close.pct_change(50).iloc[-1] if len(close) >= 51 else 0

        # 超额收益（相对大盘）
        excess_20d = 0.0
        if index_df is not None and len(index_df) >= 20:
            idx_ret20 = index_df["close"].astype(float).pct_change(20).iloc[-1]
            if not np.isnan(idx_ret20):
                excess_20d = ret20 - idx_ret20

        rows.append({
            "code": code,
            "ret_20d": ret20,
            "ret_50d": ret50,
            "excess_return_20d": excess_20d,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.set_index("code")

    # RPS = 百分位排名 * 100
    for col in ["ret_20d", "ret_50d"]:
        series = result[col].rank(pct=True) * 100
        suffix = col.replace("ret_", "")
        result[f"rps_{suffix}"] = series

    result = result.drop(columns=["ret_20d", "ret_50d"])

    print(f"[自定义因子] RPS完成，覆盖 {len(result)} 只")
    sys.stdout.flush()
    return result


# ============================================================
# 批量计算所有自定义因子
# ============================================================

# ============================================================
# 趋势强度因子（v2.0 新增：连续阳线 + 量价趋势 + 突破强度）
# ============================================================

def factor_trend_strength_single(kline_df, index_df=None) -> dict:
    """
    计算单只股票的趋势强度综合评分

    四个维度：
    1. 阳线占比：近 N 日收阳天数 / N
    2. 量价方向一致性：价格向上天数 vs 成交量放大天数的重合度
    3. 突破强度：当前价格 vs 近 20 日高点的距离
    4. 均线斜率：MA5 vs MA20 的差距（越陡越强）

    Returns:
        dict of individual factor scores (0-1 normalized)
    """
    if kline_df is None or len(kline_df) < 20:
        return {}

    close = kline_df["close"].astype(float)
    open_ = kline_df["open"].astype(float)
    high = kline_df["high"].astype(float)
    vol = kline_df["volume"].astype(float)

    N = min(10, len(close) - 1)  # 回看天数

    # 1. 阳线占比
    up_days = (close.iloc[-N:] > open_.iloc[-N:]).sum()
    up_ratio = up_days / N

    # 2. 量价方向一致性
    price_up = close.diff().iloc[-N:] > 0
    vol_up = vol.diff().iloc[-N:] > 0
    consistency = (price_up & vol_up).sum() / max(N, 1)

    # 3. 突破强度：距 20 日高点距离
    high_20 = high.iloc[-20:].max()
    latest_close = close.iloc[-1]
    breakout_pct = safe_div(latest_close - high_20, high_20)  # 负数=未突破

    # 4. 均线斜率
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    if len(ma5) > 0 and len(ma20) > 0:
        slope = safe_div(ma5.iloc[-1] - ma20.iloc[-1], ma20.iloc[-1]) * 100
    else:
        slope = 0.0

    # 5. 近 3 日涨幅
    ret_3d = safe_div(close.iloc[-1] - close.iloc[-4], close.iloc[-4]) if len(close) >= 4 else 0.0

    # 6. 相对大盘超额收益
    excess_return = 0.0
    if index_df is not None and len(index_df) > 0:
        idx_close = index_df["close"].astype(float)
        if len(idx_close) >= 2:
            idx_chg = safe_div(idx_close.iloc[-1] - idx_close.iloc[-2], idx_close.iloc[-2])
            stock_chg = safe_div(close.iloc[-1] - close.iloc[-2], close.iloc[-2])
            excess_return = stock_chg - idx_chg

    return {
        "up_day_ratio": round(up_ratio, 4),
        "vol_price_consistency": round(consistency, 4),
        "breakout_pct": round(breakout_pct, 4),
        "ma_slope_5v20": round(slope, 4),
        "ret_3d": round(ret_3d, 4),
        "excess_return": round(excess_return, 4),
    }


def compute_trend_strength_factors(kline_dict: dict, index_df=None) -> pd.DataFrame:
    """
    批量计算趋势强度因子

    Args:
        kline_dict: {code: kline_df}
        index_df: 指数日线 DataFrame

    Returns:
        DataFrame: index=code, columns=[trend_up_ratio, trend_consistency,
                  trend_breakout, trend_slope, trend_ret_3d, trend_excess]
    """
    rows = []
    for code, df in kline_dict.items():
        try:
            scores = factor_trend_strength_single(df, index_df)
            if scores:
                scores["code"] = code
                rows.append(scores)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("code")
    df.columns = [f"trend_{c}" for c in df.columns]

    # 综合趋势强度评分（等权平均）
    if len(df.columns) > 0:
        # 对每个子因子做 min-max 归一化
        df_norm = pd.DataFrame(index=df.index)
        for col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df_norm[col] = (df[col] - mn) / (mx - mn)
            else:
                df_norm[col] = 0.5

        df["trend_strength_score"] = df_norm.mean(axis=1)

    print(f"[自定义因子] 趋势强度完成，覆盖 {len(df)} 只")
    return df


def compute_all_custom_factors(
    kline_dict: dict,
    money_flow_dict: dict = None,
    index_df: pd.DataFrame = None,
    money_trend_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    计算所有自定义因子并合并

    Returns:
        DataFrame: index=code, 包含所有自定义因子列
    """
    print("[自定义因子] ===== 开始批量计算 =====")

    # 1. 量价结构因子（从kline计算）
    print("[自定义因子] 计算量价结构因子...")
    vp_rows = []
    for code, df in kline_dict.items():
        try:
            vp_rows.append({
                "code": code,
                "vol_price_corr_diff": factor_vol_price_corr_diff(df),
                "vol_price_slope": factor_vol_price_slope(df),
                "breakout_volume": factor_breakout_volume(df),
                "shrink_pullback": factor_shrink_pullback(df),
                "volume_breakout_strength": factor_volume_breakout_strength(df),
                "candle_body_ratio": factor_candle_body_ratio(df),
                "upper_shadow_ratio": factor_upper_shadow(df),
                "lower_shadow_ratio": factor_lower_shadow(df),
                "open_gap": factor_open_gap(df),
                "intraday_volatility": factor_intraday_volatility(df),
            })
        except Exception:
            continue

    df_vp = pd.DataFrame(vp_rows)
    if not df_vp.empty:
        df_vp = df_vp.set_index("code")
    print(f"[自定义因子] 量价结构完成，覆盖 {len(df_vp)} 只")

    # 2. 资金流向因子
    df_mf = pd.DataFrame()
    if money_flow_dict:
        df_mf = compute_money_flow_factors(money_flow_dict, kline_dict)

    # 3. 资金趋势因子
    if money_trend_df is not None and not money_trend_df.empty:
        df_mt = money_trend_df
    else:
        df_mt = compute_money_trend_factors(kline_dict)

    # 4. RPS因子
    df_rps = compute_rps_factors(kline_dict, index_df)

    # 5. 趋势强度因子（v2.0 新增）
    df_trend = compute_trend_strength_factors(kline_dict, index_df)

    # ===== 合并所有因子 =====
    frames = [df_vp]
    if not df_mf.empty:
        frames.append(df_mf)
    if not df_mt.empty:
        frames.append(df_mt)
    if not df_rps.empty:
        frames.append(df_rps)
    if not df_trend.empty:
        frames.append(df_trend)

    # 按code合并
    result = pd.concat(frames, axis=1, join="outer")
    result = result.loc[:, ~result.columns.duplicated()]

    print(f"[自定义因子] 合并完成，共 {len(result)} 只，{len(result.columns)} 个因子")
    sys.stdout.flush()
    return result
