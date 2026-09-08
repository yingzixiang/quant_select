"""
intraday/signal_validator.py
Section 8+9：信号校验与防骗线体系

四重校验：
1. 瞬时假突破：价格刺破轨线后快速收回，持续 < 2 根 K 线
2. 动量衰竭：连续 N 根 K 线实体逐步缩小
3. 背离风险：指数与个股走势严重背离
4. 大单异常：盘口巨量压单/托单（数据不可用则跳过）

触发任意一条 → 信号作废。
"""
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from intraday.config import INDICATORS, DATA_DEGRADATION
from intraday.signal_engine import Signal, SignalDirection
from intraday.indicators import IndicatorState, check_momentum_exhaustion

# ---- 降级日志 ----
_warnings_issued: set = set()


def _warn_once(key: str, message: str):
    if key not in _warnings_issued:
        _warnings_issued.add(key)


# ============================================================
# 校验 1：瞬时假突破
# ============================================================

def check_fake_breakout(bar_history: pd.DataFrame,
                         bar_index: int,
                         ind: IndicatorState,
                         signal_direction: str) -> Tuple[bool, str]:
    """
    检测瞬时假突破

    规则：价格短暂刺破轨线后快速收回，持续 < 2 根 K 线

    Args:
        bar_history: 完整 K 线历史
        bar_index: 当前 bar 索引
        ind: 指标快照
        signal_direction: "long"（低吸）或 "short"（高抛）

    Returns:
        (is_fake, reason)
    """
    if bar_index < 3:
        return (False, "")

    recent_bars = bar_history.iloc[bar_index - 2:bar_index + 1]
    if len(recent_bars) < 3:
        return (False, "")

    close = recent_bars["close"].astype(float)

    if signal_direction == "long":
        # 正 T 信号在缓冲区间 → 检查是否刚刺破下轨又收回
        # 前 2 根 K 线内有 bar 最低价跌破下轨，但收盘又收回
        for i in range(len(recent_bars) - 1):
            bar_low = float(recent_bars.iloc[i]["low"])
            bar_close = float(recent_bars.iloc[i]["close"])
            if bar_low < ind.lower_band and bar_close > ind.lower_band:
                # 刺破回收
                if bar_index - (bar_index - 2 + i) < 2:  # 2 根 K 线内
                    return (True, f"瞬时假突破: 最低 {bar_low:.2f} < 下轨 {ind.lower_band:.2f}，已收回")
    else:
        # 反 T 信号 → 检查是否刚刺破上轨又收回
        for i in range(len(recent_bars) - 1):
            bar_high = float(recent_bars.iloc[i]["high"])
            bar_close = float(recent_bars.iloc[i]["close"])
            if bar_high > ind.upper_band and bar_close < ind.upper_band:
                if bar_index - (bar_index - 2 + i) < 2:
                    return (True, f"瞬时假突破: 最高 {bar_high:.2f} > 上轨 {ind.upper_band:.2f}，已收回")

    return (False, "")


# ============================================================
# 校验 2：动量衰竭
# ============================================================

def check_momentum_exhaustion_validator(bar_history: pd.DataFrame,
                                         bar_index: int) -> Tuple[bool, str]:
    """
    检测动能衰竭

    规则：连续 3 根 K 线实体逐步缩小

    复用 indicators.check_momentum_exhaustion()
    """
    exhausted = check_momentum_exhaustion(bar_history.iloc[:bar_index + 1],
                                           INDICATORS["momentum_exhaustion_bars"])
    if exhausted:
        return (True, "动能衰竭: 连续 K 线实体缩小")
    return (False, "")


# ============================================================
# 校验 3：指数个股背离
# ============================================================

def check_divergence(bar_history: pd.DataFrame,
                      bar_index: int,
                      index_history: pd.DataFrame,
                      index_bar_index: int,
                      signal_direction: str) -> Tuple[bool, str]:
    """
    检测指数与个股走势背离

    规则：
    - 指数下跌，个股逆势拉涨（正 T 信号） → 非良性，降级处理（不直接作废，但标记风险）
    - 指数上涨，个股逆势下跌（反 T 信号） → 同上

    Returns:
        (is_divergence, reason)
        注意：背离不直接作废信号，但返回 True 供调用方降级处理
    """
    if index_history is None or len(index_history) < bar_index + 1:
        return (False, "指数数据不可用")

    try:
        idx_bar_index = min(index_bar_index, len(index_history) - 1)
        idx_recent = index_history.iloc[max(0, idx_bar_index - 2):idx_bar_index + 1]

        if len(idx_recent) < 2:
            return (False, "")

        idx_change = (float(idx_recent.iloc[-1]["close"]) -
                       float(idx_recent.iloc[0]["close"])) / float(idx_recent.iloc[0]["close"]) * 100

        # 背离判定
        threshold = 0.3  # 指数变化超过 0.3% 才算背离

        if signal_direction == "long" and idx_change < -threshold:
            return (True, f"指数下跌 {idx_change:.2f}%，个股逆势做多（背离风险）")
        elif signal_direction == "short" and idx_change > threshold:
            return (True, f"指数上涨 {idx_change:.2f}%，个股逆势做空（背离风险）")

    except Exception:
        pass

    return (False, "")


# ============================================================
# 综合校验
# ============================================================

def validate_signal(signal: Signal,
                    bar_history: pd.DataFrame,
                    bar_index: int,
                    ind: IndicatorState,
                    index_history: Optional[pd.DataFrame] = None,
                    index_bar_index: int = 0) -> Tuple[bool, str, dict]:
    """
    综合信号校验

    四重检查全部通过 → 信号有效，否则作废或降级。

    Returns:
        (is_valid, reason, diagnostics)
        diagnostics: {
            "fake_breakout": bool,
            "momentum_exhaustion": bool,
            "divergence": bool,
            "divergence_risk_level": "normal" | "elevated" | "high",
            "rejection_reasons": [str],
        }
    """
    diagnostics = {
        "fake_breakout": False,
        "momentum_exhaustion": False,
        "divergence": False,
        "divergence_risk_level": "normal",
        "rejection_reasons": [],
    }

    direction_str = signal.direction.value

    # 校验 1：假突破
    is_fake, fake_reason = check_fake_breakout(bar_history, bar_index, ind, direction_str)
    if is_fake:
        diagnostics["fake_breakout"] = True
        diagnostics["rejection_reasons"].append(fake_reason)
        return (False, fake_reason, diagnostics)

    # 校验 2：动量衰竭
    is_exhausted, exhaust_reason = check_momentum_exhaustion_validator(bar_history, bar_index)
    if is_exhausted:
        diagnostics["momentum_exhaustion"] = True
        diagnostics["rejection_reasons"].append(exhaust_reason)
        return (False, exhaust_reason, diagnostics)

    # 校验 3：背离检查（不直接作废，但降级）
    if index_history is not None:
        is_div, div_reason = check_divergence(
            bar_history, bar_index, index_history, index_bar_index,
            direction_str
        )
        if is_div:
            diagnostics["divergence"] = True
            diagnostics["divergence_risk_level"] = "elevated"
            # 背离不直接作废，但降低仓位（在 position_manager 中处理）
            # 此处不 return False

    # 校验 4：大单异常（数据不可用，跳过）
    # 委比极值过滤在数据不可用时跳过

    is_valid = len(diagnostics["rejection_reasons"]) == 0
    return (is_valid, "passed" if is_valid else "; ".join(diagnostics["rejection_reasons"]),
            diagnostics)


# ============================================================
# 批量校验
# ============================================================

def validate_signals_batch(signals: List[Signal],
                           stock_minute_dict: dict,
                           bar_index: int,
                           indicator_states: dict,
                           index_history: Optional[pd.DataFrame] = None,
                           index_bar_index: int = 0) -> List[Signal]:
    """
    批量校验信号，返回通过校验的信号列表

    Args:
        signals: 候选信号列表
        stock_minute_dict: {code: minute_df}
        bar_index: 当前 bar 索引
        indicator_states: {code: IndicatorState}
        index_history: 指数 K 线
        index_bar_index: 指数 bar 索引

    Returns:
        通过校验的信号列表
    """
    valid_signals = []

    for sig in signals:
        df = stock_minute_dict.get(sig.stock_code)
        ind = indicator_states.get(sig.stock_code)

        if df is None or ind is None:
            continue

        is_valid, reason, diag = validate_signal(
            sig, df, bar_index, ind, index_history, index_bar_index
        )

        if is_valid:
            # 背离风险降级：信号仍有效但降低仓位
            if diag.get("divergence_risk_level") == "elevated":
                sig.size_ratio *= 0.5  # 背离信号仓位减半
                sig.priority -= 1
            valid_signals.append(sig)

    return valid_signals
