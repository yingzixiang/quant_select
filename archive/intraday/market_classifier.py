"""
intraday/market_classifier.py
Section 3：日内行情自动分型（核心分支，策略动态切换）

四种行情类型：
- 标准震荡市（RANGE）：价格围绕 VWAP 小幅波动，均线缠绕
- 多头趋势市（BULL）：价格 > VWAP，均线多头排列
- 空头趋势市（BEAR）：价格 < VWAP，均线空头排列
- 变盘拐头市（REVERSAL）：震荡区间被放量突破/跌破

滞回机制（Hysteresis）：
- 震荡→趋势：偏离 > 0.9%，持续 >= 3 根 K 线
- 趋势→震荡：偏离 < 0.6%，持续 >= 5 根 K 线
- 切换确认：需连续 2 次复核（每 10 分钟复核一次）
"""
import sys
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Tuple, Optional

from intraday.config import MARKET_CLASSIFIER, INDICATORS
from intraday.indicators import compute_intraday_center, compute_mas, get_ma_alignment


class MarketType(IntEnum):
    """行情分型"""
    RANGE = 0      # 标准震荡市
    BULL = 1       # 多头趋势市
    BEAR = 2       # 空头趋势市
    REVERSAL = 3   # 变盘拐头市

    def __str__(self):
        names = {0: "震荡", 1: "多头", 2: "空头", 3: "变盘"}
        return names.get(self.value, "未知")


# ============================================================
# 参数
# ============================================================

HYST = MARKET_CLASSIFIER["hysteresis"]


# ============================================================
# 价格偏离度计算
# ============================================================

def compute_price_deviation(df: pd.DataFrame, bar_index: int) -> float:
    """
    计算当前价格相对于 VWAP 中枢的偏离度（%）

    Args:
        df: K 线历史
        bar_index: 当前 bar 索引

    Returns:
        偏离度百分比（正 = 价格 > 中枢，负 = 价格 < 中枢）
    """
    hist = df.iloc[:bar_index + 1]
    center_series = compute_intraday_center(hist)

    if len(center_series) < 1:
        return 0.0

    center = center_series.iloc[-1]
    price = float(hist.iloc[-1]["close"])

    if center <= 0:
        return 0.0

    return (price - center) / center * 100


# ============================================================
# 均线发散判断
# ============================================================

def check_ma_divergence(df: pd.DataFrame, bar_index: int) -> Tuple[str, bool]:
    """
    判断均线是否单向发散

    Args:
        df: K 线历史
        bar_index: 当前 bar 索引

    Returns:
        (alignment, is_diverging)
        alignment: "bull" / "bear" / "tangled"
        is_diverging: 是否单向发散（MA5-10 差值扩大）
    """
    hist = df.iloc[:bar_index + 1]

    if len(hist) < 6:
        return ("tangled", False)

    mas = compute_mas(hist)
    ma5 = mas["MA5"]
    ma10 = mas["MA10"]
    ma20 = mas["MA20"]

    current_alignment = get_ma_alignment(
        float(ma5.iloc[-1]),
        float(ma10.iloc[-1]),
        float(ma20.iloc[-1]),
    )

    if current_alignment == "tangled":
        return ("tangled", False)

    # 判断是否发散：最近 3 根 K 线的 MA 间距在扩大
    if len(ma5) >= 4:
        diff_start = abs(ma5.iloc[-4] - ma10.iloc[-4])
        diff_end = abs(ma5.iloc[-1] - ma10.iloc[-1])
        is_diverging = diff_end > diff_start * 1.05  # 间距扩大 5%
    else:
        is_diverging = False

    return (current_alignment, is_diverging)


# ============================================================
# MA20 斜率快速通道（补充价格偏离度检测不到的趋势行情）
# ============================================================

def _check_ma20_slope_trend(df: pd.DataFrame, bar_index: int) -> Tuple[Optional[MarketType], str]:
    """
    基于 MA20 斜率的趋势快速检测

    当 MA20 持续向同一方向倾斜时，即使价格偏离 VWAP 不大，
    也判定为趋势市。解决单边慢跌/慢涨行情中分型器滞后的
    问题（如回测中 002396 连跌 19% 仍判震荡）。

    Args:
        df: K 线历史
        bar_index: 当前 bar 索引

    Returns:
        (MarketType or None, diagnostic)
        None 表示 MA20 斜率不足以判定趋势
    """
    cfg = MARKET_CLASSIFIER.get("ma20_slope_fast_path", {})
    if not cfg.get("enabled", True):
        return (None, "")

    hist = df.iloc[:bar_index + 1]
    slope_bars = cfg["slope_bars"]
    threshold = cfg["slope_threshold_pct"]
    consecutive = cfg["consecutive_bars"]

    if len(hist) < slope_bars + consecutive:
        return (None, "数据不足")

    mas = compute_mas(hist)
    ma20 = mas["MA20"]

    if len(ma20) < slope_bars + consecutive:
        return (None, "")

    # 计算最近 consecutive 根 bar 的 MA20 斜率
    # 斜率 = (当前MA20 - N根前MA20) / N / 当前MA20 * 100 (%)
    up_count = 0
    down_count = 0

    for offset in range(consecutive):
        i = len(ma20) - 1 - offset
        j = i - slope_bars
        if j < 0:
            continue
        ma_now = float(ma20.iloc[i])
        ma_past = float(ma20.iloc[j])
        if ma_now <= 0:
            continue
        slope = (ma_now - ma_past) / slope_bars / ma_now * 100  # % per bar
        if slope > threshold:
            up_count += 1
        elif slope < -threshold:
            down_count += 1

    total = up_count + down_count
    if total == 0:
        return (None, "MA20 斜率不显著")

    # 超过 70% 的检查点同向 → 快速通道触发
    if total >= consecutive * 0.7:
        if up_count > down_count:
            return (MarketType.BULL, f"MA20 斜率快速通道: 多头 ({up_count}/{total})")
        elif down_count > up_count:
            return (MarketType.BEAR, f"MA20 斜率快速通道: 空头 ({down_count}/{total})")

    return (None, "")


# ============================================================
# 核心：滞回行情分型
# ============================================================

def classify_market_with_hysteresis(
    df: pd.DataFrame,
    bar_index: int,
    prev_type: MarketType,
    confirm_count: int = 0,
) -> Tuple[MarketType, int, str]:
    """
    带滞回区间的行情分型（状态机）

    两条检测路径（任一触发即可）：
    A. 价格偏离度路径：价格偏离 VWAP > 阈值 + 均线发散
    B. MA20 斜率快速通道：MA20 持续单向倾斜

    切换到新分型需要：
    1. 满足新分型的条件
    2. 连续 confirm_required 次复核时都指向新分型

    Args:
        df: K 线历史
        bar_index: 当前 bar 索引
        prev_type: 上一次的分型结果
        confirm_count: 连续指向新分型的次数

    Returns:
        (new_type, new_confirm_count, diagnostic_reason)
    """
    hist = df.iloc[:bar_index + 1]

    if len(hist) < 10:
        return (MarketType.RANGE, 0, "数据不足，默认震荡")

    deviation = compute_price_deviation(df, bar_index)
    alignment, is_diverging = check_ma_divergence(df, bar_index)
    cur_price = float(hist.iloc[-1]["close"])
    volume = float(hist.iloc[-1].get("volume", 0))

    # ---- 路径 B: MA20 斜率快速通道 ----
    fast_type, fast_diag = _check_ma20_slope_trend(df, bar_index)

    # ---- 路径 A: 价格偏离度路径 ----
    if _is_reversal(df, bar_index):
        deviation_type = MarketType.REVERSAL
    elif abs(deviation) <= HYST["trend_to_range_price_pct"]:
        deviation_type = MarketType.RANGE
    elif deviation > HYST["range_to_trend_price_pct"] and alignment == "bull" and is_diverging:
        deviation_type = MarketType.BULL
    elif deviation < -HYST["range_to_trend_price_pct"] and alignment == "bear" and is_diverging:
        deviation_type = MarketType.BEAR
    else:
        deviation_type = None  # 无法判定

    # ---- 综合两条路径确定原始分型 ----
    # 优先变盘信号
    if deviation_type == MarketType.REVERSAL:
        raw_type = MarketType.REVERSAL
    # MA20 快速通道优先于震荡（解决单边慢行情的分型滞后）
    elif fast_type is not None and deviation_type in (None, MarketType.RANGE):
        raw_type = fast_type
    # 价格偏离路径
    elif deviation_type is not None:
        raw_type = deviation_type
    else:
        raw_type = MarketType.RANGE

    # ---- 滞回状态机 ----
    if raw_type == prev_type:
        return (prev_type, 0, f"维持{prev_type}")

    # 原始分型与上次不同——需要确认
    new_confirm = confirm_count + 1

    if new_confirm >= HYST["confirm_required"]:
        # 确认切换
        return (raw_type, 0, f"切换: {prev_type} → {raw_type}（确认 {new_confirm} 次）")
    else:
        # 尚未确认，保持原分型
        return (prev_type, new_confirm,
                f"候选 {raw_type}，待确认 ({new_confirm}/{HYST['confirm_required']})")


# ============================================================
# 变盘检测
# ============================================================

def _is_reversal(df: pd.DataFrame, bar_index: int) -> bool:
    """
    判断是否进入变盘拐头市

    条件：
    - 原有震荡区间被放量突破/跌破
    - 量比 >= 1.8
    - 均线由缠绕转为单向发散

    Args:
        df: K 线历史
        bar_index: 当前 bar 索引
    """
    hist = df.iloc[:bar_index + 1]

    if len(hist) < 15:
        return False

    close = hist["close"].astype(float)
    volume = hist["volume"].astype(float)

    # 量比：当前量 / 近 10 根均量
    recent_vol_avg = volume.iloc[-11:-1].mean() if len(volume) >= 11 else volume.iloc[:-1].mean()
    if recent_vol_avg <= 0:
        return False

    vol_ratio = volume.iloc[-1] / recent_vol_avg
    if vol_ratio < MARKET_CLASSIFIER["reversal_volume_ratio"]:
        return False

    # 检查是否突破震荡区间
    # 用前 15 根 K 线的最高/最低作为震荡区间
    prev_15 = hist.iloc[-16:-1]  # 不含当前 bar
    if len(prev_15) < 10:
        return False

    range_high = prev_15["high"].astype(float).max()
    range_low = prev_15["low"].astype(float).min()
    range_size = range_high - range_low

    if range_size <= 0:
        return False

    # 当前价格突破区间 30% 以上
    cur_high = float(hist.iloc[-1]["high"])
    cur_low = float(hist.iloc[-1]["low"])

    break_up = cur_high > range_high and (cur_high - range_high) / range_size > 0.3
    break_down = cur_low < range_low and (range_low - cur_low) / range_size > 0.3

    # 均线是否开始发散
    alignment, is_diverging = check_ma_divergence(df, bar_index)

    return (break_up or break_down) and alignment != "tangled" and is_diverging


# ============================================================
# 初始分型（09:35-10:05 初次判断）
# ============================================================

def initial_classify(df: pd.DataFrame, center_window_end_idx: int) -> Tuple[MarketType, str]:
    """
    中枢窗口结束时的初始分型

    Args:
        df: K 线历史
        center_window_end_idx: 中枢窗口结束的 bar index

    Returns:
        (market_type, diagnostic)
    """
    deviation = compute_price_deviation(df, center_window_end_idx)
    alignment, is_diverging = check_ma_divergence(df, center_window_end_idx)

    if _is_reversal(df, center_window_end_idx):
        return (MarketType.REVERSAL, "变盘信号")

    abs_dev = abs(deviation)

    if abs_dev <= HYST["trend_to_range_price_pct"]:
        return (MarketType.RANGE, f"偏离 {deviation:.2f}%（震荡区间）")

    if deviation > HYST["range_to_trend_price_pct"] and alignment == "bull":
        return (MarketType.BULL, f"偏离 {deviation:+.2f}%，多头排列")

    if deviation < -HYST["range_to_trend_price_pct"] and alignment == "bear":
        return (MarketType.BEAR, f"偏离 {deviation:+.2f}%，空头排列")

    return (MarketType.RANGE, f"偏离 {deviation:+.2f}%，默认震荡")


# ============================================================
# 分型 → 允许的交易类型
# ============================================================

def get_allowed_directions(market_type: MarketType) -> dict:
    """
    返回该行情下允许的交易方向

    Returns:
        {"allow_long": bool, "allow_short": bool, "max_rounds": int}
    """
    if market_type == MarketType.RANGE:
        return {"allow_long": True, "allow_short": True, "max_rounds": 3}
    elif market_type == MarketType.BULL:
        return {"allow_long": True, "allow_short": False, "max_rounds": 3}
    elif market_type == MarketType.BEAR:
        return {"allow_long": False, "allow_short": True, "max_rounds": 3}
    elif market_type == MarketType.REVERSAL:
        return {"allow_long": True, "allow_short": True, "max_rounds": 1}
    else:
        return {"allow_long": False, "allow_short": False, "max_rounds": 0}
