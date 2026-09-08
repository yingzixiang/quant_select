"""
intraday/risk_manager.py
Section 7：全域硬风控体系

五级风控：
1. 价格风控（涨跌停）
2. 交易频次风控
3. 分级亏损风控（单笔 → 单标的 → 全账户）
4. 滑点 & 委托风控
5. 突发风险风控

所有规则为硬性触发，无人工豁免。
"""
import sys
from enum import IntEnum
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from intraday.config import RISK


class RiskAction(IntEnum):
    """风控动作"""
    HOLD = 0                 # 继续持有
    STOP_OUT = 1             # 硬止损
    TAKE_PROFIT_TIER1 = 2    # 阶梯止盈①（部分卖出）
    TAKE_PROFIT_TIER2 = 3    # 阶梯止盈②（剩余清仓）
    TAKE_PROFIT_FULL = 4     # 合并全仓止盈
    REVERSE_RETREAT = 5      # 反向撤离
    TRAILING_STOP = 6        # 移动止盈（趋势市）
    TIME_CLOSE = 7           # 时间平仓（14:40 强制）
    COVER_RETURN = 8         # 接回（反 T 场景）

class OrderAction(IntEnum):
    """委托动作"""
    KEEP = 0
    CANCEL = 1
    ADJUST = 2


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TradeRecord:
    """单笔交易记录"""
    stock_code: str
    direction: str            # "long" / "short"
    open_time: str            # HH:MM
    open_price: float
    shares: int
    stop_loss_price: float
    take_profit_tiers: List[Tuple[float, float]] = field(default_factory=list)  # [(price, ratio)]
    reverse_retreat_atr: float = 0.0
    scenario: str = "range"
    # 动态更新
    high_price: float = 0.0   # 持仓期间最高价（用于移动止盈）
    low_price: float = 999999 # 持仓期间最低价
    partial_closed: bool = False  # 阶梯止盈①是否已触发


# ============================================================
# RiskManager
# ============================================================

class RiskManager:
    """单标的日内风控管理器"""

    def __init__(self, code: str):
        self.code = code
        self.frequency_pauses: Dict[str, str] = {}  # {direction: pause_until_time}
        self.same_direction_failures: Dict[str, int] = {"long": 0, "short": 0}
        self.total_trades_today: int = 0
        self.price_limit_locked: bool = False
        self.lock_reason: str = ""

    # ---- 价格风控 ----

    def check_price_limit(self,
                          cur_price: float,
                          prev_close: float,
                          is_limit_up: bool = False,
                          is_limit_down: bool = False) -> RiskAction:
        """
        检查涨跌停风控

        Args:
            cur_price: 当前价格
            prev_close: 前日收盘价
            is_limit_up: 是否封死涨停
            is_limit_down: 是否封死跌停

        Returns:
            HOLD / STOP_OUT
        """
        if is_limit_up:
            self.price_limit_locked = True
            self.lock_reason = "涨停封死"
            return RiskAction.STOP_OUT

        if is_limit_down:
            self.price_limit_locked = True
            self.lock_reason = "跌停封死"
            return RiskAction.STOP_OUT

        # 盘中瞬时涨跌停反复打开
        change_pct = (cur_price - prev_close) / prev_close * 100
        if abs(change_pct) >= 9.5:  # 接近涨跌停
            self.price_limit_locked = True
            self.lock_reason = f"接近涨跌停 {change_pct:+.1f}%"

        return RiskAction.HOLD

    # ---- 交易频次风控 ----

    def check_frequency_limit(self, direction: str,
                              current_time: str) -> Tuple[bool, str]:
        """
        检查是否处于信号暂停期

        规则：同一方向信号连续触发 2 次未成交，暂停该信号 15 分钟

        Returns:
            (blocked, reason)
        """
        pause_until = self.frequency_pauses.get(direction, "")
        if pause_until and current_time < pause_until:
            return (True, f"信号暂停至 {pause_until}")
        return (False, "")

    def record_signal_failure(self, direction: str, current_time: str):
        """记录信号未成交"""
        self.same_direction_failures[direction] += 1

        if self.same_direction_failures[direction] >= RISK["signal_pause_threshold"]:
            # 计算暂停到何时
            parts = current_time.split(":")
            pause_min = int(parts[0]) * 60 + int(parts[1]) + RISK["signal_pause_minutes"]
            pause_h = min(pause_min // 60, 14)
            pause_m = min(pause_min % 60, 59)
            pause_until = f"{pause_h:02d}:{pause_m:02d}"
            self.frequency_pauses[direction] = pause_until

    def record_signal_executed(self, direction: str):
        """信号成交，重置计数器"""
        self.same_direction_failures[direction] = 0
        if direction in self.frequency_pauses:
            del self.frequency_pauses[direction]

    # ---- 分级亏损风控 ----

    def check_stop_loss(self,
                        trade: TradeRecord,
                        cur_bar: dict,
                        indicator_state) -> Tuple[RiskAction, str]:
        """
        检查单笔交易的止损/止盈条件

        多层检查顺序：
        1. 硬止损（价格跌破/突破止损线 或 单笔亏损达阈值）
        2. 反向撤离（不利方向超 0.3 ATR）
        3. 阶梯止盈（①触碰上轨 ②盈利达阈值）
        4. 移动止盈（趋势市，跌破 MA5）

        Args:
            trade: 交易记录
            cur_bar: 当前 bar 的 OHLCV 数据
            indicator_state: 当前指标快照

        Returns:
            (action, reason)
        """
        cur_price = float(cur_bar["close"])
        cur_high = float(cur_bar.get("high", cur_price))
        cur_low = float(cur_bar.get("low", cur_price))

        # 更新极值
        trade.high_price = max(trade.high_price, cur_high)
        trade.low_price = min(trade.low_price, cur_low)

        direction = trade.direction
        entry_price = trade.open_price

        # 当前浮盈百分比
        if direction == "long":
            pnl_pct = (cur_price - entry_price) / entry_price * 100
            adverse_excursion = (entry_price - trade.low_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - cur_price) / entry_price * 100
            adverse_excursion = (trade.high_price - entry_price) / entry_price * 100

        # ---- 1. 硬止损 ----
        if direction == "long":
            stop_triggered = (cur_low <= trade.stop_loss_price)
        else:
            stop_triggered = (cur_high >= trade.stop_loss_price)

        if stop_triggered:
            return (RiskAction.STOP_OUT, f"价格触及止损线 ({trade.stop_loss_price:.2f})")

        # 百分比止损
        from intraday.config import SCENARIO_RULES
        stop_pct = SCENARIO_RULES["range_trading"]["hard_stop_pct"]
        if pnl_pct <= -stop_pct:
            return (RiskAction.STOP_OUT, f"单笔亏损 {pnl_pct:.2f}% >= {stop_pct}%")

        # ---- 2. 反向撤离 ----
        if adverse_excursion >= trade.reverse_retreat_atr * 100:
            return (RiskAction.REVERSE_RETREAT, f"反向运行 {adverse_excursion:.2f}% > {trade.reverse_retreat_atr:.1f} ATR")

        # ---- 3. 阶梯止盈（场景一：震荡市） ----
        if trade.scenario == "range":
            if trade.take_profit_tiers:
                tier1_price, tier1_ratio = trade.take_profit_tiers[0]

                if direction == "long" and cur_high >= tier1_price and not trade.partial_closed:
                    if len(trade.take_profit_tiers) > 1:
                        trade.partial_closed = True
                        return (RiskAction.TAKE_PROFIT_TIER1,
                                f"阶梯止盈①: 触碰上轨 {tier1_price:.2f}，卖出 {tier1_ratio*100:.0f}%")
                    else:
                        return (RiskAction.TAKE_PROFIT_FULL, "阶梯止盈（合并）")

                # 冲突处理：两档同时触发
                if trade.partial_closed and pnl_pct >= SCENARIO_RULES["range_trading"]["staggered_tp_2_pct"]:
                    return (RiskAction.TAKE_PROFIT_TIER2, f"阶梯止盈②: 盈利 {pnl_pct:.2f}%")

                # 如果阶梯①触发时盈利已达止盈②阈值，合并执行
                if (not trade.partial_closed and
                    cur_high >= tier1_price and
                    pnl_pct >= SCENARIO_RULES["range_trading"]["staggered_tp_2_pct"]):
                    return (RiskAction.TAKE_PROFIT_FULL, f"合并止盈: 触碰上轨 + 盈利 {pnl_pct:.2f}%")

        # ---- 4. 移动止盈（场景二/三：趋势市） ----
        if trade.scenario in ("bull", "bear"):
            # 跌破 MA5 → 全部卖出
            if direction == "long" and cur_low <= indicator_state.ma5:
                return (RiskAction.TRAILING_STOP, f"移动止盈: 跌破 MA5 ({indicator_state.ma5:.2f})")
            elif direction == "short" and cur_high >= indicator_state.ma5:
                return (RiskAction.TRAILING_STOP, f"移动止盈: 突破 MA5 ({indicator_state.ma5:.2f})")

        return (RiskAction.HOLD, "")

    # ---- 滑点 & 委托风控 ----

    def get_execution_price(self, target_price: float, is_buy: bool) -> float:
        """
        计算含滑点的执行价格

        Args:
            target_price: 目标价格
            is_buy: 是否买入

        Returns:
            实际执行价格
        """
        offset = RISK["slippage_offset"]
        if is_buy:
            return target_price + offset  # 买入加价保证成交
        else:
            return target_price - offset  # 卖出减价保证成交

    # ---- 重置 ----

    def reset_day(self):
        """重置当日状态"""
        self.frequency_pauses.clear()
        self.same_direction_failures = {"long": 0, "short": 0}
        self.total_trades_today = 0
        self.price_limit_locked = False
        self.lock_reason = ""


# ============================================================
# 辅助：时间比较
# ============================================================

def time_before(t1: str, t2: str) -> bool:
    """判断 t1 是否在 t2 之前（HH:MM 格式）"""
    return t1 < t2


def is_close_only_time(timestamp_str: str) -> bool:
    """判断当前是否仅可平仓（14:40 后）"""
    t = timestamp_str[:5]  # HH:MM
    return t >= "14:40" and t < "14:57"


def should_cancel_all(timestamp_str: str) -> bool:
    """尾盘集合竞价前是否应撤销所有委托"""
    t = timestamp_str[:5]
    return t >= "14:57"
