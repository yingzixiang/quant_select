"""
intraday/indicators.py
自适应指标体系（Section 4）：纯计算函数，无状态

覆盖：
- 动态价格区间（ATR + 日内中枢）
- 趋势指标（MA5/MA10/MA20 组合）
- 量能指标（相对量比、拐点量、突破量）
- K 线形态过滤（影线、十字星、动能衰竭）
- 盘口微观指标（价差、BS 比率、大单监控）

所有函数接受 bar_history（截至当前 bar 的 OHLCV 历史）并返回计算结果。
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field

from intraday.config import INDICATORS, DATA_DEGRADATION

# ---- 全局降级日志 ----
_warnings_issued: set = set()


def _warn_once(key: str, message: str):
    if key not in _warnings_issued:
        _warnings_issued.add(key)


# ============================================================
# 核心数据结构
# ============================================================

@dataclass
class IndicatorState:
    """单个 bar 时刻的完整指标快照"""
    timestamp: str = ""
    # 价格区间
    center: float = 0.0              # VWAP 日内中枢
    atr: float = 0.0                 # 真实波幅 ATR(20)
    upper_band: float = 0.0          # 动态上轨 = center + multiplier * ATR
    lower_band: float = 0.0          # 动态下轨 = center - multiplier * ATR
    upper_buffer_start: float = 0.0  # 上轨缓冲区间起点
    upper_buffer_end: float = 0.0    # 上轨缓冲区间终点
    lower_buffer_start: float = 0.0  # 下轨缓冲区间起点
    lower_buffer_end: float = 0.0    # 下轨缓冲区间终点
    # 趋势
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma_alignment: str = "tangled"    # "bull" / "bear" / "tangled"
    ma20_direction: str = "flat"     # "up" / "down" / "flat"
    # 量能
    relative_volume: float = 0.0     # 相对量比
    volume_trend: str = "flat"       # "increasing" / "decreasing" / "flat"
    prev_3min_avg_vol: float = 0.0   # 前 3 分钟均量（5min bar 时近似为前 1 根）
    # K 线形态
    candle_body_ratio: float = 0.0   # 实体/振幅
    candle_upper_shadow_ratio: float = 0.0  # 上影/振幅
    candle_lower_shadow_ratio: float = 0.0  # 下影/振幅
    is_doji: bool = False            # 十字星
    # 盘口
    bs_ratio: Optional[float] = None  # 主动买卖比（数据不可用时为 None）
    price_deviation_pct: float = 0.0  # 价格偏离中枢百分比
    amplitude: float = 0.0            # 当前 bar 振幅


# ============================================================
# 1. 动态价格区间
# ============================================================

def compute_atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    计算 ATR（真实波幅）

    ATR = MA(TR, window)
    TR = max(H-L, |H-C_prev|, |L-C_prev|)

    Args:
        df: 需含 high, low, close 列
        window: 平滑窗口

    Returns:
        pd.Series of ATR values
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window, min_periods=1).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    计算累积 VWAP（成交量加权均价）

    VWAP = Σ(price * volume) / Σ(volume)

    使用 (H+L+C)/3 作为典型价格。

    Args:
        df: 需含 high, low, close, volume 列

    Returns:
        pd.Series of cumulative VWAP
    """
    typical_price = (df["high"].astype(float) +
                     df["low"].astype(float) +
                     df["close"].astype(float)) / 3
    volume = df["volume"].astype(float)

    cum_vp = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()

    vwap = cum_vp / cum_vol.replace(0, np.nan)
    return vwap.fillna(typical_price)


def compute_intraday_center(df: pd.DataFrame,
                            center_window_bars: int = 30,
                            period_minutes: int = 5) -> pd.Series:
    """
    计算日内动态中枢

    规则：
    - 09:35~10:05（或 30 根 1min bar / 6 根 5min bar）：累积 VWAP
    - 之后：EMA 滚动更新

    Args:
        df: 分钟数据
        center_window_bars: 中枢窗口 K 线数
        period_minutes: 每根 K 线分钟数

    Returns:
        pd.Series of dynamic center values
    """
    vwap = compute_vwap(df)
    center = vwap.copy()

    # 中枢窗口之后用 EMA 更新
    alpha = 2 / (center_window_bars + 1)
    for i in range(center_window_bars, len(center)):
        center.iloc[i] = center.iloc[i - 1] * (1 - alpha) + vwap.iloc[i] * alpha

    return center


def compute_dynamic_bands(center: float,
                          atr: float,
                          multiplier: float = 1.5,
                          buffer: float = 0.3) -> Dict:
    """
    计算动态上下轨及缓冲区间

    Args:
        center: 日内中枢
        atr: 当前 ATR 值
        multiplier: ATR 倍数（默认 1.5）
        buffer: 缓冲区间 ATR 倍数（默认 0.3）

    Returns:
        {
            "upper_band": float,
            "lower_band": float,
            "upper_buffer_start": float,
            "upper_buffer_end": float,
            "lower_buffer_start": float,
            "lower_buffer_end": float,
        }
    """
    upper = center + multiplier * atr
    lower = center - multiplier * atr
    buf = buffer * atr

    return {
        "upper_band": upper,
        "lower_band": lower,
        "upper_buffer_start": upper - buf,
        "upper_buffer_end": upper,
        "lower_buffer_start": lower,
        "lower_buffer_end": lower + buf,
    }


# ============================================================
# 2. 趋势指标
# ============================================================

def compute_mas(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    计算 MA5, MA10, MA20

    Args:
        df: 需含 close 列

    Returns:
        {"MA5": Series, "MA10": Series, "MA20": Series}
    """
    close = df["close"].astype(float)

    return {
        "MA5": close.rolling(5, min_periods=1).mean(),
        "MA10": close.rolling(10, min_periods=1).mean(),
        "MA20": close.rolling(20, min_periods=1).mean(),
    }


def get_ma_alignment(ma5: float, ma10: float, ma20: float) -> str:
    """判断均线排列状态"""
    if ma5 > ma10 > ma20:
        return "bull"
    elif ma5 < ma10 < ma20:
        return "bear"
    else:
        return "tangled"


def get_ma20_direction(ma20_series: pd.Series, lookback: int = 5) -> str:
    """
    判断 MA20 方向

    Args:
        ma20_series: MA20 序列
        lookback: 回看 K 线数
    """
    if len(ma20_series) < lookback + 1:
        return "flat"

    recent = ma20_series.iloc[-lookback - 1:]
    if len(recent) < 2:
        return "flat"

    slope = recent.iloc[-1] - recent.iloc[0]
    threshold = recent.iloc[0] * 0.001  # 0.1% 阈值避免微幅波动误判

    if slope > threshold:
        return "up"
    elif slope < -threshold:
        return "down"
    else:
        return "flat"


# ============================================================
# 3. 量能指标
# ============================================================

def compute_relative_volume(current_vol: float,
                            ref_vol: Optional[float]) -> float:
    """
    计算相对量比：当前量 / 历史同时段均量

    Args:
        current_vol: 当前 bar 成交量
        ref_vol: 历史同时段均量（来自 volume_profile）

    Returns:
        量比值，ref_vol 为 None 或 0 时返回 1.0
    """
    if ref_vol is None or ref_vol <= 0:
        return 1.0
    return current_vol / ref_vol


def compute_volume_trend(df: pd.DataFrame, lookback: int = 3) -> str:
    """
    判断近 N 根 K 线的量能趋势

    Args:
        df: 需含 volume 列
        lookback: 回看 K 线数
    """
    if len(df) < lookback:
        return "flat"

    vol = df["volume"].astype(float).iloc[-lookback:]
    first_half = vol.iloc[:lookback // 2 + 1].mean()
    second_half = vol.iloc[-lookback // 2 - 1:].mean()

    if second_half > first_half * 1.1:
        return "increasing"
    elif second_half < first_half * 0.9:
        return "decreasing"
    else:
        return "flat"


# ============================================================
# 4. K 线形态过滤
# ============================================================

def detect_candle_patterns(bar: pd.DataFrame) -> Dict:
    """
    检测单根 K 线形态

    Args:
        bar: 单行 DataFrame，需含 open, high, low, close

    Returns:
        {
            "body_ratio": float,      # 实体占振幅比例
            "upper_shadow_ratio": float,
            "lower_shadow_ratio": float,
            "is_doji": bool,          # 十字星（实体/振幅 < 10%）
            "body_direction": str,    # "up" / "down" / "flat"
        }
    """
    o = bar["open"]
    h = bar["high"]
    l = bar["low"]
    c = bar["close"]

    hl_range = h - l
    if hl_range <= 0:
        return {
            "body_ratio": 0, "upper_shadow_ratio": 0,
            "lower_shadow_ratio": 0, "is_doji": True, "body_direction": "flat"
        }

    body = abs(c - o)
    body_high = max(o, c)
    body_low = min(o, c)

    upper_shadow = h - body_high
    lower_shadow = body_low - l

    result = {
        "body_ratio": body / hl_range,
        "upper_shadow_ratio": upper_shadow / hl_range,
        "lower_shadow_ratio": lower_shadow / hl_range,
        "is_doji": body / hl_range < 0.1,
        "body_direction": "up" if c > o else ("down" if c < o else "flat"),
    }
    return result


def check_momentum_exhaustion(df: pd.DataFrame, lookback: int = 3) -> bool:
    """
    检测动能衰竭：连续 N 根 K 线实体逐步缩小

    Args:
        df: K 线 DataFrame
        lookback: 回看 K 线数

    Returns:
        True 表示动能衰竭
    """
    if len(df) < lookback + 1:
        return False

    recent = df.iloc[-lookback - 1:]
    bodies = []
    for _, bar in recent.iterrows():
        body = abs(bar["close"] - bar["open"])
        hl = bar["high"] - bar["low"]
        bodies.append(body / hl if hl > 0 else 0)

    # 检查最后 N 根是否递减
    last_n = bodies[-lookback:]
    for i in range(1, len(last_n)):
        if last_n[i] >= last_n[i - 1]:
            return False

    return True


# ============================================================
# 5. 盘口微观指标（优雅降级）
# ============================================================

def check_spread_valid(spread: Optional[float], threshold: float = 0.03) -> Tuple[bool, str]:
    """
    检查五档价差是否在可接受范围

    Returns:
        (passed, reason)
    """
    if spread is None:
        return (True, "order_book_unavailable")
    if spread > threshold:
        return (False, f"spread_{spread:.3f}_too_wide")
    return (True, "spread_ok")


# ============================================================
# 6. 综合指标计算
# ============================================================

def compute_indicator_state(df: pd.DataFrame,
                            bar_index: int,
                            vol_ref: Optional[float] = None,
                            atr_multiplier: float = 1.5,
                            atr_buffer: float = 0.5,
                            atr_window: int = 20) -> IndicatorState:
    """
    计算某一 bar 时刻的完整指标快照

    这是前端的聚合函数，在回测循环中每个 bar 调用一次。

    Args:
        df: 截至当前 bar 的完整 K 线历史
        bar_index: 当前 bar 的索引位置
        vol_ref: 历史同时段均量（来自 volume_profile），None 则用量比 1.0
        atr_multiplier: ATR 倍数
        atr_buffer: 缓冲区 ATR 倍数
        atr_window: ATR 平滑窗口

    Returns:
        IndicatorState
    """
    state = IndicatorState()

    if len(df) < 3:
        return state

    # 切片到当前 bar（不含未来数据）
    hist = df.iloc[:bar_index + 1]
    cur_bar = hist.iloc[-1]
    cur_price = float(cur_bar["close"])

    # ---- 时间戳 ----
    if "timestamp" in df.columns:
        ts = df.iloc[bar_index]["timestamp"]
        state.timestamp = str(ts)

    # ---- 价格区间 ----
    atr_series = compute_atr(hist, atr_window)
    state.atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0

    center_series = compute_intraday_center(hist)
    state.center = float(center_series.iloc[-1]) if len(center_series) > 0 else cur_price

    if state.center > 0:
        state.price_deviation_pct = (cur_price - state.center) / state.center * 100

    bands = compute_dynamic_bands(state.center, state.atr, atr_multiplier, atr_buffer)
    state.upper_band = bands["upper_band"]
    state.lower_band = bands["lower_band"]
    state.upper_buffer_start = bands["upper_buffer_start"]
    state.upper_buffer_end = bands["upper_buffer_end"]
    state.lower_buffer_start = bands["lower_buffer_start"]
    state.lower_buffer_end = bands["lower_buffer_end"]

    # ---- 趋势 ----
    mas = compute_mas(hist)
    state.ma5 = float(mas["MA5"].iloc[-1]) if len(mas["MA5"]) > 0 else cur_price
    state.ma10 = float(mas["MA10"].iloc[-1]) if len(mas["MA10"]) > 0 else cur_price
    state.ma20 = float(mas["MA20"].iloc[-1]) if len(mas["MA20"]) > 0 else cur_price
    state.ma_alignment = get_ma_alignment(state.ma5, state.ma10, state.ma20)
    state.ma20_direction = get_ma20_direction(mas["MA20"])

    # ---- 量能 ----
    cur_vol = float(cur_bar.get("volume", 0))
    state.relative_volume = compute_relative_volume(cur_vol, vol_ref)
    state.volume_trend = compute_volume_trend(hist)
    # 前 3 分钟均量（5min bar 近似为前 1 根）
    if len(hist) >= 2:
        state.prev_3min_avg_vol = float(hist["volume"].astype(float).iloc[-2])
    else:
        state.prev_3min_avg_vol = cur_vol

    # ---- K 线形态 ----
    patterns = detect_candle_patterns(cur_bar)
    state.candle_body_ratio = patterns["body_ratio"]
    state.candle_upper_shadow_ratio = patterns["upper_shadow_ratio"]
    state.candle_lower_shadow_ratio = patterns["lower_shadow_ratio"]
    state.is_doji = patterns["is_doji"]
    state.amplitude = float(cur_bar["high"] - cur_bar["low"])

    # ---- 盘口 ----
    # BS 比率在 5 分钟 bar 中不可用
    state.bs_ratio = None

    return state


# ============================================================
# 辅助：获取中枢窗口结束的 bar index
# ============================================================

def get_center_window_end_index(df: pd.DataFrame,
                                period_minutes: int = 5) -> int:
    """
    根据时间戳找到 10:05 对应的 bar index

    对于 5 分钟 K 线：09:35-10:05 包含 09:35, 09:40, 09:45, 09:50, 09:55, 10:00, 10:05
    共 7 根（index 0-6），返回 6

    Args:
        df: 分钟数据
        period_minutes: 每根 K 线分钟数

    Returns:
        bar index
    """
    if "timestamp" not in df.columns:
        # 根据 period 估算
        return max(30 // period_minutes, 3)

    target = pd.Timestamp("10:05:00").time()
    for i, ts in enumerate(df["timestamp"]):
        if ts.time() >= target:
            return i

    return min(len(df) // 10, 10)  # fallback


# ============================================================
# 辅助：按 bar 时间判断当前处于哪个阶段
# ============================================================

def get_time_phase(timestamp_str: str) -> str:
    """
    判断当前 bar 所处的时间阶段

    Returns:
        "pre_trading"  (09:30-09:35) → 禁止交易
        "center_window" (09:35-10:05) → 中枢建立期
        "normal"        (10:05-14:40) → 正常交易
        "close_only"    (14:40-14:57) → 仅平仓
        "closing_auction" (14:57-15:00) → 集合竞价
    """
    try:
        t = pd.Timestamp(timestamp_str).time()
    except Exception:
        return "normal"

    if pd.Timestamp("09:30:00").time() <= t < pd.Timestamp("09:35:00").time():
        return "pre_trading"
    elif pd.Timestamp("09:35:00").time() <= t < pd.Timestamp("10:05:00").time():
        return "center_window"
    elif pd.Timestamp("10:05:00").time() <= t < pd.Timestamp("14:40:00").time():
        return "normal"
    elif pd.Timestamp("14:40:00").time() <= t < pd.Timestamp("14:57:00").time():
        return "close_only"
    else:
        return "closing_auction"
