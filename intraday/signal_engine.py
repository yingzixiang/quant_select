"""
intraday/signal_engine.py
Section 5：分场景完整交易规则

五种场景的信号生成：
- 场景一：标准震荡市（双向 T）
- 场景二：多头趋势市（仅正 T）
- 场景三：空头趋势市（仅反 T）
- 场景四：变盘拐头市（单次顺势）
- 场景五：跳空缺口行情

每个场景的正 T/反 T 入场规则为全部条件 AND。
"""
import sys
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

from intraday.config import SCENARIO_RULES, INDICATORS
from intraday.market_classifier import MarketType
from intraday.indicators import IndicatorState


class SignalDirection(Enum):
    """信号方向"""
    LONG = "long"    # 正 T（低吸 → 高抛）
    SHORT = "short"  # 反 T（高抛 → 低吸）


class Scenario(Enum):
    """交易场景"""
    RANGE = "range"        # 场景一
    BULL = "bull"          # 场景二
    BEAR = "bear"          # 场景三
    REVERSAL = "reversal"  # 场景四
    GAP = "gap"            # 场景五


@dataclass
class Signal:
    """交易信号"""
    stock_code: str
    timestamp: str
    direction: SignalDirection
    scenario: Scenario
    entry_price: float
    size_shares: int = 0
    size_ratio: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_tiers: List[Tuple[float, float]] = field(default_factory=list)
    reverse_retreat_atr: float = 0.0
    conditions_met: List[str] = field(default_factory=list)
    conditions_failed: List[str] = field(default_factory=list)
    priority: int = 0


# ============================================================
# 条件检查辅助
# ============================================================

def _check(condition: bool, name: str,
           met: List[str], failed: List[str]) -> bool:
    """记录条件结果"""
    if condition:
        met.append(name)
    else:
        failed.append(name)
    return condition


# ============================================================
# 场景一：标准震荡市 — 正 T（低吸 → 高抛）
# ============================================================

def signals_range_long(stock_code: str,
                        timestamp: str,
                        ind: IndicatorState,
                        cur_bar: dict,
                        gap_scenario: int = 0) -> Optional[Signal]:
    """
    震荡市正 T 入场信号

    全部条件 AND：
    1. 价格在「下轨 ~ 下轨 + 0.3ATR」缓冲区间
    2. MA10 上穿 MA20（金叉），MA20 走平或微向上
    3. 相对量比 >= 1.1，无明显缩量
    4. K 线实体饱满，无长下影
    5. 环境无风险（调用方保证）
    """
    met, failed = [], []
    cur_price = float(cur_bar["close"])

    # 条件 1：价格位置
    in_buffer = (ind.lower_buffer_start <= cur_price <= ind.lower_buffer_end)
    _check(in_buffer, f"价格在缓冲区间 [{ind.lower_buffer_start:.2f}, {ind.lower_buffer_end:.2f}]",
           met, failed)

    # 条件 2：MA 交叉与方向
    # MA10 在 MA20 之上且 MA20 方向不为 down
    ma_cross_ok = (ind.ma_alignment != "bear" and ind.ma20_direction != "down")
    _check(ma_cross_ok, f"MA排列={ind.ma_alignment}, MA20方向={ind.ma20_direction}", met, failed)

    # 条件 3：量能（v3.3: 量比提高到 1.15 且需量能递增）
    min_vol_ratio = INDICATORS.get("min_relative_volume_long", 1.15)
    vol_ok = (ind.relative_volume >= min_vol_ratio)
    _check(vol_ok, f"量比={ind.relative_volume:.2f} (需≥{min_vol_ratio})", met, failed)

    if INDICATORS.get("require_volume_increasing", True):
        vol_up = (ind.volume_trend == "increasing")
        _check(vol_up, f"量能趋势={ind.volume_trend}", met, failed)

    # 条件 4：K 线形态
    no_long_lower = (ind.candle_lower_shadow_ratio <= INDICATORS["candle_lower_shadow_ratio"])
    _check(no_long_lower, f"下影比={ind.candle_lower_shadow_ratio:.2f}", met, failed)

    body_ok = (ind.candle_body_ratio > 0.2 and not ind.is_doji)
    _check(body_ok, f"实体比={ind.candle_body_ratio:.2f}, is_doji={ind.is_doji}", met, failed)

    # 条件 5：K 线方向确认（v3.3: 做多需收阳）
    if INDICATORS.get("require_candle_direction", True):
        cur_open = float(cur_bar.get("open", cur_price))
        candle_up = (cur_price >= cur_open)
        _check(candle_up, f"K线方向={'阳' if candle_up else '阴'}", met, failed)

    if len(failed) > 0:
        return None

    # ---- 构建信号 ----
    signal = Signal(
        stock_code=stock_code,
        timestamp=timestamp,
        direction=SignalDirection.LONG,
        scenario=Scenario.RANGE,
        entry_price=cur_price,
        conditions_met=met,
        conditions_failed=failed,
    )

    # 阶梯止盈
    rules = SCENARIO_RULES["range_trading"]
    signal.take_profit_tiers = [
        (ind.upper_band, rules["staggered_tp_1_ratio"]),
        (signal.entry_price * (1 + rules["staggered_tp_2_pct"] / 100), 1.0),
    ]

    # 硬止损
    signal.stop_loss_price = ind.lower_band - rules["hard_stop_atr"] * ind.atr
    if signal.stop_loss_price <= 0:
        signal.stop_loss_price = cur_price * (1 - rules["hard_stop_pct"] / 100)

    # 反向撤离阈值
    signal.reverse_retreat_atr = rules["reverse_retreat_atr"] * ind.atr

    return signal


# ============================================================
# 场景一：标准震荡市 — 反 T（高抛 → 低吸）
# ============================================================

def signals_range_short(stock_code: str,
                         timestamp: str,
                         ind: IndicatorState,
                         cur_bar: dict,
                         gap_scenario: int = 0) -> Optional[Signal]:
    """
    震荡市反 T 入场信号

    全部条件 AND：
    1. 价格在「上轨 ~ 上轨 - 0.3ATR」缓冲区间
    2. MA10 下穿 MA20（死叉），MA20 走平或微向下
    3. 冲高过程量能逐步萎缩
    4. K 线无长上影诱多
    """
    met, failed = [], []
    cur_price = float(cur_bar["close"])

    # 条件 1：价格位置
    in_buffer = (ind.upper_buffer_start <= cur_price <= ind.upper_buffer_end)
    _check(in_buffer, f"价格在缓冲区间 [{ind.upper_buffer_start:.2f}, {ind.upper_buffer_end:.2f}]",
           met, failed)

    # 条件 2：MA 交叉与方向
    ma_cross_ok = (ind.ma_alignment != "bull" and ind.ma20_direction != "up")
    _check(ma_cross_ok, f"MA排列={ind.ma_alignment}, MA20方向={ind.ma20_direction}", met, failed)

    # 条件 3：量能萎缩
    vol_shrinking = (ind.volume_trend == "decreasing" or ind.relative_volume < 1.1)
    _check(vol_shrinking, f"量能趋势={ind.volume_trend}, 量比={ind.relative_volume:.2f}", met, failed)

    # 条件 4：无长上影
    no_long_upper = (ind.candle_upper_shadow_ratio <= INDICATORS["candle_upper_shadow_ratio"])
    _check(no_long_upper, f"上影比={ind.candle_upper_shadow_ratio:.2f}", met, failed)

    body_ok = (ind.candle_body_ratio > 0.2 and not ind.is_doji)
    _check(body_ok, f"实体比={ind.candle_body_ratio:.2f}", met, failed)

    # 条件 5：K 线方向确认（v3.3: 做空需收阴）
    if INDICATORS.get("require_candle_direction", True):
        cur_open = float(cur_bar.get("open", cur_price))
        candle_down = (cur_price < cur_open)
        _check(candle_down, f"K线方向={'阴' if candle_down else '阳'}", met, failed)

    if len(failed) > 0:
        return None

    # ---- 构建信号 ----
    signal = Signal(
        stock_code=stock_code,
        timestamp=timestamp,
        direction=SignalDirection.SHORT,
        scenario=Scenario.RANGE,
        entry_price=cur_price,
        conditions_met=met,
        conditions_failed=failed,
    )

    rules = SCENARIO_RULES["range_trading"]
    signal.stop_loss_price = ind.upper_band + rules["breakout_cover_atr"] * ind.atr
    signal.take_profit_tiers = [
        (ind.lower_band, 1.0),  # 回落至下轨接回
    ]
    signal.reverse_retreat_atr = rules["reverse_retreat_atr"] * ind.atr

    return signal


# ============================================================
# 场景二：多头趋势市（仅正 T）
# ============================================================

def signals_bull_long(stock_code: str,
                       timestamp: str,
                       ind: IndicatorState,
                       cur_bar: dict) -> Optional[Signal]:
    """
    多头趋势市正 T 入场

    入场：价格回调至 MA10 + VWAP 双重支撑位
    止盈：移动止盈（跌破 MA5）
    止损：跌破均价 0.6%
    """
    met, failed = [], []
    cur_price = float(cur_bar["close"])
    rules = SCENARIO_RULES["bull_trend"]

    # 条件 1：价格回调至支撑区间
    # MA10 和 VWAP 双重支撑
    support_lower = min(ind.ma10, ind.center) * 0.995  # 支撑下浮 0.5%
    support_upper = max(ind.ma10, ind.center) * 1.005
    near_support = (support_lower <= cur_price <= support_upper)
    _check(near_support, f"价格={cur_price:.2f} 在支撑区间 [{support_lower:.2f}, {support_upper:.2f}]",
           met, failed)

    # 条件 2：均线保持多头排列
    bull_alignment = (ind.ma_alignment == "bull")
    _check(bull_alignment, f"MA排列={ind.ma_alignment}", met, failed)

    # 条件 3：量能温和（不缩量）
    vol_ok = (ind.relative_volume >= 0.8)
    _check(vol_ok, f"量比={ind.relative_volume:.2f}", met, failed)

    if len(failed) > 0:
        return None

    signal = Signal(
        stock_code=stock_code,
        timestamp=timestamp,
        direction=SignalDirection.LONG,
        scenario=Scenario.BULL,
        entry_price=cur_price,
        conditions_met=met,
        conditions_failed=failed,
    )

    # 止盈：移动止盈（跌破 MA5）
    signal.take_profit_tiers = []  # 不预设固定目标
    signal.stop_loss_price = ind.center * (1 - rules["stop_loss_pct"] / 100)
    signal.reverse_retreat_atr = 0.5 * ind.atr

    return signal


# ============================================================
# 场景三：空头趋势市（仅反 T）
# ============================================================

def signals_bear_short(stock_code: str,
                        timestamp: str,
                        ind: IndicatorState,
                        cur_bar: dict) -> Optional[Signal]:
    """
    空头趋势市反 T 入场

    入场：价格反弹至 MA10 + VWAP 双重压力位
    接回：回落 0.9%
    止损：突破近 20 根 K 线阶段高点
    """
    met, failed = [], []
    cur_price = float(cur_bar["close"])
    rules = SCENARIO_RULES["bear_trend"]

    # 条件 1：价格反弹至压力区间
    resistance_lower = min(ind.ma10, ind.center) * 0.995
    resistance_upper = max(ind.ma10, ind.center) * 1.005
    near_resistance = (resistance_lower <= cur_price <= resistance_upper)
    _check(near_resistance, f"价格={cur_price:.2f} 在压力区间 [{resistance_lower:.2f}, {resistance_upper:.2f}]",
           met, failed)

    # 条件 2：均线空头排列
    bear_alignment = (ind.ma_alignment == "bear")
    _check(bear_alignment, f"MA排列={ind.ma_alignment}", met, failed)

    # 条件 3：反弹量能不足
    vol_weak = (ind.relative_volume < 1.2)
    _check(vol_weak, f"量比={ind.relative_volume:.2f}", met, failed)

    if len(failed) > 0:
        return None

    signal = Signal(
        stock_code=stock_code,
        timestamp=timestamp,
        direction=SignalDirection.SHORT,
        scenario=Scenario.BEAR,
        entry_price=cur_price,
        conditions_met=met,
        conditions_failed=failed,
    )

    # 接回：回落 0.9%
    signal.take_profit_tiers = [
        (cur_price * (1 - rules["cover_drop_pct"] / 100), 1.0),
    ]
    # 止损：突破阶段高点（由 risk_manager 在 bar 循环中动态更新）
    signal.stop_loss_price = cur_price * 1.02  # 临时 2%，由 bar_history 最高价动态更新
    signal.reverse_retreat_atr = 0.5 * ind.atr

    return signal


# ============================================================
# 场景四：变盘拐头市（单次顺势）
# ============================================================

def signals_reversal(stock_code: str,
                      timestamp: str,
                      ind: IndicatorState,
                      cur_bar: dict,
                      bar_history: pd.DataFrame) -> Optional[Signal]:
    """
    变盘拐头市交易信号

    - 区间放量上破：仅做一次低吸
    - 区间放量下破：仅做一次高抛
    """
    met, failed = [], []
    cur_price = float(cur_bar["close"])
    cur_volume = float(cur_bar.get("volume", 0))
    rules = SCENARIO_RULES["reversal"]

    if len(bar_history) < rules["short_term_bars"]:
        return None

    prev_bars = bar_history.iloc[-rules["short_term_bars"] - 1:-1]
    prev_high = prev_bars["high"].astype(float).max()
    prev_low = prev_bars["low"].astype(float).min()
    prev_vol_avg = prev_bars["volume"].astype(float).mean()

    # 判断上破还是下破
    break_up = cur_price > prev_high and cur_volume > prev_vol_avg * 1.5
    break_down = cur_price < prev_low and cur_volume > prev_vol_avg * 1.5

    if break_up:
        # 上破 → 低吸，目标短期高点
        short_term_target = prev_bars["high"].astype(float).max() + rules["pressure_atr"] * ind.atr
        signal = Signal(
            stock_code=stock_code,
            timestamp=timestamp,
            direction=SignalDirection.LONG,
            scenario=Scenario.REVERSAL,
            entry_price=cur_price,
            conditions_met=met + ["放量上破"],
            priority=1,
        )
        signal.take_profit_tiers = [(short_term_target, 1.0)]
        signal.stop_loss_price = prev_high * 0.99  # 跌破前高 1%
        signal.reverse_retreat_atr = 0.5 * ind.atr
        return signal

    if break_down:
        # 下破 → 高抛，回落后接回
        cover_zone = ind.lower_band - rules["cover_zone_atr"] * ind.atr
        signal = Signal(
            stock_code=stock_code,
            timestamp=timestamp,
            direction=SignalDirection.SHORT,
            scenario=Scenario.REVERSAL,
            entry_price=cur_price,
            conditions_met=met + ["放量下破"],
            priority=1,
        )
        signal.take_profit_tiers = [(cover_zone, 1.0)]
        signal.stop_loss_price = prev_low * 1.01  # 突破前低 1%
        signal.reverse_retreat_atr = 0.5 * ind.atr
        return signal

    return None


# ============================================================
# 场景五：跳空缺口行情
# ============================================================

def signals_gap(stock_code: str,
                timestamp: str,
                ind: IndicatorState,
                cur_bar: dict,
                gap_pct: float,
                gap_scenario: int,
                day_open: float) -> Optional[Signal]:
    """
    跳空缺口行情交易信号

    方向：回补缺口方向
    仓位：按跳空幅度分档

    Args:
        gap_pct: 跳空幅度（可正可负）
        gap_scenario: 0=无, 1=小, 2=中, 3=大
        day_open: 当日开盘价
    """
    if gap_scenario == 0 or gap_scenario == 3:
        return None  # 无跳空或跳空过大不交易

    met, failed = [], []
    cur_price = float(cur_bar["close"])
    rules = SCENARIO_RULES["gap"]

    # 方向：回补缺口
    if gap_pct > 0:
        # 高开 → 做反 T（高抛 → 低吸回补）
        direction = SignalDirection.SHORT
        scenario_spec = Scenario.GAP
    else:
        # 低开 → 做正 T（低吸 → 高抛回补）
        direction = SignalDirection.LONG
        scenario_spec = Scenario.GAP

    # 入场条件简化：价格偏离开盘价向不利方向运行后
    deviation_from_open = (cur_price - day_open) / day_open * 100

    if direction == SignalDirection.LONG:
        # 低开后不再创新低、有止跌迹象
        in_zone = (cur_price <= day_open * 1.005 and cur_price >= day_open * 0.985)
    else:
        # 高开后不再创新高、有滞涨迹象
        in_zone = (cur_price >= day_open * 0.995 and cur_price <= day_open * 1.015)

    _check(in_zone, f"价格在缺口回补区 ({cur_price:.2f} vs 开盘 {day_open:.2f})", met, failed)

    # 量价健康
    vol_ok = (ind.relative_volume >= 0.8)
    _check(vol_ok, f"量比={ind.relative_volume:.2f}", met, failed)

    if len(failed) > 0:
        return None

    signal = Signal(
        stock_code=stock_code,
        timestamp=timestamp,
        direction=direction,
        scenario=Scenario.GAP,
        entry_price=cur_price,
        conditions_met=met,
        conditions_failed=failed,
        priority=2,  # 缺口信号优先级高于震荡
    )

    # 止盈：缺口回补至距离开盘价 ±0.15%
    close_target = day_open * (1 + (0.15 / 100 if gap_pct > 0 else -0.15 / 100))
    signal.take_profit_tiers = [(close_target, 1.0)]

    # 止损：反向突破开盘价 0.5%
    if direction == SignalDirection.LONG:
        signal.stop_loss_price = day_open * (1 - rules["sl_gap_reverse_pct"] / 100)
    else:
        signal.stop_loss_price = day_open * (1 + rules["sl_gap_reverse_pct"] / 100)

    signal.reverse_retreat_atr = 0.2 * ind.atr

    return signal


# ============================================================
# 信号调度器
# ============================================================

def generate_signals(stock_code: str,
                     timestamp: str,
                     ind: IndicatorState,
                     cur_bar: dict,
                     market_type: MarketType,
                     gap_scenario: int = 0,
                     gap_pct: float = 0.0,
                     day_open: float = 0.0,
                     bar_history: pd.DataFrame = None,
                     env_blocked: bool = False,
                     stock_blocked: bool = False) -> List[Signal]:
    """
    信号生成调度器

    根据当前行情分型和跳空状态，路由到对应的场景函数。

    Args:
        stock_code: 股票代码
        timestamp: 当前 bar 时间戳
        ind: 指标快照
        cur_bar: 当前 bar 数据
        market_type: 行情分型
        gap_scenario: 跳空级别
        gap_pct: 跳空幅度
        day_open: 当日开盘价
        bar_history: 完整 K 线历史
        env_blocked: 全局环境是否封堵
        stock_blocked: 个股是否被封堵

    Returns:
        List[Signal]（最多 2 个，震荡市双向各一个）
    """
    if env_blocked or stock_blocked:
        return []

    signals = []

    # ---- 缺口行情优先 ----
    if gap_scenario in (1, 2) and gap_pct != 0:
        gap_sig = signals_gap(stock_code, timestamp, ind, cur_bar,
                              gap_pct, gap_scenario, day_open)
        if gap_sig:
            # 缺口场景只做缺口方向，不做其他场景
            return [gap_sig]

    # ---- 变盘行情 ----
    if market_type == MarketType.REVERSAL:
        rev_sig = signals_reversal(stock_code, timestamp, ind, cur_bar, bar_history)
        return [rev_sig] if rev_sig else []

    # ---- 趋势行情 ----
    if market_type == MarketType.BULL:
        sig = signals_bull_long(stock_code, timestamp, ind, cur_bar)
        return [sig] if sig else []

    if market_type == MarketType.BEAR:
        sig = signals_bear_short(stock_code, timestamp, ind, cur_bar)
        return [sig] if sig else []

    # ---- 震荡行情（双向） ----
    if market_type == MarketType.RANGE:
        long_sig = signals_range_long(stock_code, timestamp, ind, cur_bar)
        short_sig = signals_range_short(stock_code, timestamp, ind, cur_bar)

        if long_sig:
            signals.append(long_sig)
        if short_sig:
            signals.append(short_sig)

    return signals
