"""
Alpha158 风格因子计算：40+ 核心量价因子
基于 Alpha158/Alpha191 中短线 IC 最高的因子精选实现

每只股票独立计算，输入为单只股票的日线 DataFrame，
输出为该股票的各因子值（series 或标量）。
"""
import numpy as np
import pandas as pd


def safe_div(a, b, fallback=0.0):
    """安全除法"""
    if b is None or b == 0:
        return fallback
    return a / b


def rolling_corr(x, y, window):
    """滚动相关系数"""
    return x.rolling(window).corr(y)


# ============================================================
# 动量类因子（Momentum）
# ============================================================

def factor_mom_5d(close):
    """5日动量"""
    return close.pct_change(5).iloc[-1]


def factor_mom_10d(close):
    """10日动量"""
    return close.pct_change(10).iloc[-1]


def factor_mom_20d(close):
    """20日动量"""
    return close.pct_change(20).iloc[-1]


def factor_rsi_6d(close):
    """6日RSI"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(6).mean()
    avg_loss = loss.rolling(6).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).iloc[-1]


def factor_rsi_12d(close):
    """12日RSI"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(12).mean()
    avg_loss = loss.rolling(12).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).iloc[-1]


def factor_macd(close):
    """MACD指标（DIF - DEA）"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = 2 * (dif - dea)
    return float(dif.iloc[-1] - dea.iloc[-1])


def factor_kdj_k(high, low, close):
    """KDJ K值"""
    n = 9
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    return float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0


def factor_williams_r(high, low, close):
    """威廉指标 WR(10)"""
    n = 10
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    wr = (hh - close) / (hh - ll).replace(0, np.nan) * -100
    return float(wr.iloc[-1]) if not pd.isna(wr.iloc[-1]) else -50.0


def factor_cci(high, low, close):
    """商品通道指数 CCI(14)"""
    n = 14
    tp = (high + low + close) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean())
    cci_val = (tp - sma) / (0.015 * mad)
    return float(cci_val.iloc[-1]) if not pd.isna(cci_val.iloc[-1]) else 0.0


# ============================================================
# 反转类因子（Reversal）
# ============================================================

def factor_reversal_3d(close):
    """3日反转：最近3日收益与之前12日收益的关系（Alpha23简化）"""
    ret3 = close.pct_change(3).iloc[-1]
    ret12_shift = close.pct_change(12).shift(3).iloc[-1]
    if ret12_shift is None or pd.isna(ret12_shift) or ret12_shift == 0:
        return 0.0
    return ret3 / abs(ret12_shift)


def factor_intraday_reversal(open_, close):
    """日内反转：今日开盘到收盘的收益"""
    return (close.iloc[-1] - open_.iloc[-1]) / open_.iloc[-1]


def factor_gap_return(open_, close):
    """跳空收益率 (今日开盘 / 昨日收盘 - 1)"""
    if len(open_) < 2 or len(close) < 2:
        return 0.0
    return open_.iloc[-1] / close.iloc[-2] - 1


def factor_reversal_5d_20d(close):
    """5日涨跌幅 vs 20日涨跌幅"""
    ret5 = close.pct_change(5).iloc[-1]
    ret20 = close.pct_change(20).iloc[-1]
    return ret5 - ret20


# ============================================================
# 波动类因子（Volatility）
# ============================================================

def factor_volatility_10d(close):
    """10日波动率"""
    ret = close.pct_change().dropna().tail(10)
    return float(np.std(ret))


def factor_volatility_20d(close):
    """20日波动率"""
    ret = close.pct_change().dropna().tail(20)
    return float(np.std(ret))


def factor_atr_14d(high, low, close):
    """ATR(14)"""
    n = 14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1]) / float(close.iloc[-1])


def factor_bollinger_position(close):
    """布林带位置：(close - lower) / (upper - lower)"""
    n = 20
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    pos = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    return float(pos) if not pd.isna(pos) else 0.5


def factor_bollinger_width(close):
    """布林带宽度：(upper - lower) / MA"""
    n = 20
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    width = (4 * std.iloc[-1]) / ma.iloc[-1]
    return float(width) if not pd.isna(width) else 0.0


# ============================================================
# 量价相关类因子（Price-Volume Correlation）
# ============================================================

def factor_corr_close_vol_5d(close, volume):
    """5日收盘价与成交量相关系数"""
    ret = close.pct_change()
    vol_chg = volume.pct_change()
    common = pd.DataFrame({"ret": ret, "vol": vol_chg}).dropna().tail(5)
    if len(common) < 3:
        return 0.0
    corr = common["ret"].corr(common["vol"])
    return float(corr) if not pd.isna(corr) else 0.0


def factor_corr_close_vol_20d(close, volume):
    """20日收盘价与成交量相关系数"""
    ret = close.pct_change()
    vol_chg = volume.pct_change()
    common = pd.DataFrame({"ret": ret, "vol": vol_chg}).dropna().tail(20)
    if len(common) < 5:
        return 0.0
    corr = common["ret"].corr(common["vol"])
    return float(corr) if not pd.isna(corr) else 0.0


def factor_volume_ratio_5d(volume):
    """5日量比：今日成交 / 5日均量"""
    ma5 = volume.rolling(5).mean()
    return float(volume.iloc[-1] / ma5.iloc[-1]) if ma5.iloc[-1] > 0 else 0.0


def factor_volume_ratio_20d(volume):
    """20日量比：今日成交 / 20日均量"""
    ma20 = volume.rolling(20).mean()
    return float(volume.iloc[-1] / ma20.iloc[-1]) if ma20.iloc[-1] > 0 else 0.0


def factor_volume_trend_5d(volume):
    """5日成交量趋势：最近5日 vs 前5日"""
    if len(volume) < 10:
        return 0.0
    recent = volume.tail(5).mean()
    prior = volume.tail(10).head(5).mean()
    return float(recent / prior) if prior > 0 else 1.0


def factor_amount_ratio(close, volume):
    """成交额变化率：5日均成交额 / 20日均成交额"""
    amount = close * volume
    ma5 = amount.rolling(5).mean()
    ma20 = amount.rolling(20).mean()
    ratio = ma5.iloc[-1] / ma20.iloc[-1]
    return float(ratio) if not pd.isna(ratio) and ma20.iloc[-1] > 0 else 0.0


# ============================================================
# 均线类因子（Moving Average）
# ============================================================

def factor_ma5_deviation(close):
    """股价与MA5的偏离度"""
    ma5 = close.rolling(5).mean().iloc[-1]
    return (close.iloc[-1] - ma5) / ma5


def factor_ma10_deviation(close):
    """股价与MA10的偏离度"""
    ma10 = close.rolling(10).mean().iloc[-1]
    return (close.iloc[-1] - ma10) / ma10


def factor_ma20_deviation(close):
    """股价与MA20的偏离度"""
    ma20 = close.rolling(20).mean().iloc[-1]
    return (close.iloc[-1] - ma20) / ma20


def factor_ma5_ma10_dist(close):
    """MA5与MA10的距离"""
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    if ma10 == 0:
        return 0.0
    return (ma5 - ma10) / ma10


def factor_ma5_ma20_dist(close):
    """MA5与MA20的距离"""
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    if ma20 == 0:
        return 0.0
    return (ma5 - ma20) / ma20


def factor_ma_dispertion(close):
    """均线发散度：各均线标准差的归一化"""
    mas = [
        close.rolling(5).mean().iloc[-1],
        close.rolling(10).mean().iloc[-1],
        close.rolling(20).mean().iloc[-1],
        close.rolling(60).mean().iloc[-1],
    ]
    mas = [m for m in mas if not pd.isna(m)]
    if len(mas) < 2:
        return 0.0
    avg = np.mean(mas)
    if avg == 0:
        return 0.0
    return np.std(mas) / avg


def factor_ma_bullish_alignment(close):
    """
    均线多头排列程度：MA5>MA10>MA20>MA60 的满足程度
    返回 0~1，1表示完美多头排列
    """
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]

    if pd.isna(ma60):
        return 0.0

    score = 0
    score += 1 if close.iloc[-1] > ma5 else 0
    score += 1 if ma5 > ma10 else 0
    score += 1 if ma10 > ma20 else 0
    score += 1 if ma20 > ma60 else 0
    return score / 4.0


# ============================================================
# 价格形态因子（Price Pattern）
# ============================================================

def factor_intraday_strength(open_, high, low, close):
    """日内阳线实体占比 (close-open)/(high-low)"""
    denom = high.iloc[-1] - low.iloc[-1]
    if denom == 0:
        return 0.0
    return (close.iloc[-1] - open_.iloc[-1]) / denom


def factor_upper_shadow_ratio(open_, high, low, close):
    """上影线占比 (high - max(open,close)) / (high - low)"""
    denom = high.iloc[-1] - low.iloc[-1]
    if abs(denom) < 1e-6:
        return 0.0
    body_high = max(open_.iloc[-1], close.iloc[-1])
    return (high.iloc[-1] - body_high) / denom


def factor_high_low_range(high, low, close):
    """(high-low)/close 日内振幅"""
    return (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]


def factor_price_position_20d(high, low, close):
    """当前价格在20日高低区间的相对位置"""
    hh = high.rolling(20).max().iloc[-1]
    ll = low.rolling(20).min().iloc[-1]
    if hh == ll:
        return 0.5
    return (close.iloc[-1] - ll) / (hh - ll)


def factor_new_high_20d(close):
    """是否为20日新高（0或1的强度）"""
    hh = close.rolling(20).max().iloc[-2]  # 前20日最高
    if pd.isna(hh) or hh == 0:
        return 0.0
    return 1.0 if close.iloc[-1] > hh else 0.0


def factor_consecutive_up_days(close):
    """连续上涨天数"""
    days = 0
    for i in range(len(close) - 1, 0, -1):
        if close.iloc[i] > close.iloc[i - 1]:
            days += 1
        else:
            break
    return min(days, 10)


def factor_consecutive_down_days(close):
    """连续下跌天数"""
    days = 0
    for i in range(len(close) - 1, 0, -1):
        if close.iloc[i] < close.iloc[i - 1]:
            days += 1
        else:
            break
    return min(days, 10)


# ============================================================
# 收益分布因子（Return Distribution）
# ============================================================

def factor_skewness_20d(close):
    """20日收益率偏度"""
    ret = close.pct_change().dropna().tail(20)
    if len(ret) < 5:
        return 0.0
    return float(ret.skew())


def factor_kurtosis_20d(close):
    """20日收益率峰度"""
    ret = close.pct_change().dropna().tail(20)
    if len(ret) < 5:
        return 0.0
    return float(ret.kurtosis())


def factor_max_drawdown_20d(close):
    """20日内最大回撤"""
    if len(close) < 20:
        return 0.0
    recent = close.tail(20)
    peak = recent.cummax()
    dd = (recent - peak) / peak
    return float(dd.min())


def factor_up_days_ratio_20d(close):
    """20日内上涨天数占比"""
    ret = close.pct_change().dropna().tail(20)
    if len(ret) == 0:
        return 0.0
    return float((ret > 0).sum() / len(ret))


def factor_avg_up_return_20d(close):
    """20日内平均上涨幅度"""
    ret = close.pct_change().dropna().tail(20)
    up = ret[ret > 0]
    if len(up) == 0:
        return 0.0
    return float(up.mean())


def factor_avg_down_return_20d(close):
    """20日内平均下跌幅度"""
    ret = close.pct_change().dropna().tail(20)
    down = ret[ret < 0]
    if len(down) == 0:
        return 0.0
    return float(down.mean())


# ============================================================
# Alpha191 经典因子（简化的短线因子）
# ============================================================

def factor_alpha1(open_, close, volume):
    """
    Alpha1简化: -corr(rank(delta(log(volume),1)), rank((close-open)/open), 6)
    量变动与日内收益的负相关
    """
    log_vol = np.log(volume.replace(0, np.nan))
    vol_delta = log_vol.diff()
    intraday_ret = (close - open_) / open_.replace(0, np.nan)

    common = pd.DataFrame({"dvol": vol_delta, "iret": intraday_ret}).dropna().tail(6)
    if len(common) < 3:
        return 0.0
    corr = common["dvol"].rank().corr(common["iret"].rank())
    return float(-corr) if not pd.isna(corr) else 0.0


def factor_alpha2(open_, high, low, close):
    """
    Alpha2简化: -delta(((close-low)-(high-close))/(high-low), 1)
    上影线与下影线之差的负变化
    """
    denom = (high - low).replace(0, np.nan)
    shadow_ratio = ((close - low) - (high - close)) / denom
    delta_val = -shadow_ratio.diff().iloc[-1]
    return float(delta_val) if not pd.isna(delta_val) else 0.0


def factor_alpha3(close):
    """
    Alpha3相关: close是否连续不变（用于衡量流动性差/停牌风险）
    sum(close==delay(close,1)?0:close<delay(close,1)?...,6) 变体
    """
    eq = (close.diff() == 0).astype(int)
    return float(eq.tail(6).sum())  # 越小越好


def factor_alpha101(open_, high, low, close, volume):
    """
    Alpha101简化: (close-open)/(high-low) + delay(close-open)/(high-low) * 0.5
    日内实体强度 + 惯性
    """
    denom = (high - low).replace(0, np.nan)
    intensity = (close - open_) / denom
    if len(intensity) < 2:
        return float(intensity.iloc[-1]) if not pd.isna(intensity.iloc[-1]) else 0.0
    return float(intensity.iloc[-1] + 0.5 * intensity.iloc[-2])


# ============================================================
# 批量计算入口
# ============================================================

_FACTOR_DEFINITIONS = [
    # (factor_name, function, required_cols)
    ("mom_5d", factor_mom_5d, ["close"]),
    ("mom_10d", factor_mom_10d, ["close"]),
    ("mom_20d", factor_mom_20d, ["close"]),
    ("rsi_6d", factor_rsi_6d, ["close"]),
    ("rsi_12d", factor_rsi_12d, ["close"]),
    ("macd", factor_macd, ["close"]),
    ("kdj_k", factor_kdj_k, ["high", "low", "close"]),
    ("williams_r", factor_williams_r, ["high", "low", "close"]),
    ("cci", factor_cci, ["high", "low", "close"]),
    ("reversal_3d", factor_reversal_3d, ["close"]),
    ("intraday_reversal", factor_intraday_reversal, ["open", "close"]),
    ("gap_return", factor_gap_return, ["open", "close"]),
    ("reversal_5d_20d", factor_reversal_5d_20d, ["close"]),
    ("volatility_10d", factor_volatility_10d, ["close"]),
    ("volatility_20d", factor_volatility_20d, ["close"]),
    ("atr_14d", factor_atr_14d, ["high", "low", "close"]),
    ("bollinger_position", factor_bollinger_position, ["close"]),
    ("bollinger_width", factor_bollinger_width, ["close"]),
    ("corr_close_vol_5d", factor_corr_close_vol_5d, ["close", "volume"]),
    ("corr_close_vol_20d", factor_corr_close_vol_20d, ["close", "volume"]),
    ("volume_ratio_5d", factor_volume_ratio_5d, ["volume"]),
    ("volume_ratio_20d", factor_volume_ratio_20d, ["volume"]),
    ("volume_trend_5d", factor_volume_trend_5d, ["volume"]),
    ("amount_ratio", factor_amount_ratio, ["close", "volume"]),
    ("ma5_deviation", factor_ma5_deviation, ["close"]),
    ("ma10_deviation", factor_ma10_deviation, ["close"]),
    ("ma20_deviation", factor_ma20_deviation, ["close"]),
    ("ma5_ma10_dist", factor_ma5_ma10_dist, ["close"]),
    ("ma5_ma20_dist", factor_ma5_ma20_dist, ["close"]),
    ("ma_dispersion", factor_ma_dispertion, ["close"]),
    ("ma_bullish", factor_ma_bullish_alignment, ["close"]),
    ("intraday_strength", factor_intraday_strength, ["open", "high", "low", "close"]),
    ("high_low_range", factor_high_low_range, ["high", "low", "close"]),
    ("price_position_20d", factor_price_position_20d, ["high", "low", "close"]),
    ("new_high_20d", factor_new_high_20d, ["close"]),
    ("consecutive_up", factor_consecutive_up_days, ["close"]),
    ("consecutive_down", factor_consecutive_down_days, ["close"]),
    ("skewness_20d", factor_skewness_20d, ["close"]),
    ("kurtosis_20d", factor_kurtosis_20d, ["close"]),
    ("max_drawdown_20d", factor_max_drawdown_20d, ["close"]),
    ("up_days_ratio", factor_up_days_ratio_20d, ["close"]),
    ("avg_up_return", factor_avg_up_return_20d, ["close"]),
    ("avg_down_return", factor_avg_down_return_20d, ["close"]),
    ("alpha1", factor_alpha1, ["open", "close", "volume"]),
    ("alpha2", factor_alpha2, ["open", "high", "low", "close"]),
    ("alpha3", factor_alpha3, ["close"]),
    ("alpha101", factor_alpha101, ["open", "high", "low", "close", "volume"]),
]


def compute_alpha_factors_single(kline_df: pd.DataFrame) -> dict:
    """
    计算单只股票的所有Alpha因子

    Args:
        kline_df: 单只股票日线DataFrame，需包含 open, high, low, close, volume 列
                  按日期升序排列，至少60条

    Returns:
        dict: {factor_name: value}，计算失败的因子值为 NaN
    """
    if kline_df is None or len(kline_df) < 60:
        return {}

    factors = {}
    for name, func, cols in _FACTOR_DEFINITIONS:
        try:
            args = [kline_df[col] for col in cols]
            val = func(*args)
            factors[name] = float(val) if not (val is None or (isinstance(val, float) and np.isnan(val))) else np.nan
        except Exception:
            factors[name] = np.nan

    return factors


def compute_alpha_factors_batch(kline_dict: dict) -> pd.DataFrame:
    """
    批量计算所有股票的Alpha因子

    Args:
        kline_dict: {code: kline_DataFrame}

    Returns:
        DataFrame: index=code, columns=各因子名
    """
    import sys

    print(f"[Alpha因子] 批量计算，共 {len(kline_dict)} 只...")
    all_factors = {}
    completed = 0

    for code, df in kline_dict.items():
        factors = compute_alpha_factors_single(df)
        if factors:
            all_factors[code] = factors
        completed += 1
        if completed % 200 == 0:
            print(f"[Alpha因子] 进度: {completed}/{len(kline_dict)}")
            sys.stdout.flush()

    result = pd.DataFrame.from_dict(all_factors, orient="index")
    result.index.name = "code"
    print(f"[Alpha因子] 完成，有效 {len(result)} 只，{len(result.columns)} 个因子")
    sys.stdout.flush()
    return result
