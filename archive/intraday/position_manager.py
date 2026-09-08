"""
intraday/position_manager.py
Section 6：动态仓位管理

功能：
- 根据行情强度、市场环境、当日盈亏动态调整做 T 仓位比例
- 仓位堆叠/轮次限制检查
- 全账户动态止损（基于当日盈亏状态）
"""
import sys
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from intraday.config import POSITION, ACCOUNT_STOP_LOSS
from intraday.market_classifier import MarketType


# ============================================================
# 数据结构
# ============================================================

@dataclass
class PositionState:
    """单标的仓位状态"""
    code: str
    base_shares: int = 0             # 底仓股数
    open_long_trade: bool = False    # 是否持有正 T 仓位
    open_short_trade: bool = False   # 是否持有反 T 仓位
    daily_rounds: int = 0            # 当日已完成轮次
    daily_pnl_pct: float = 0.0       # 当日累计盈亏（%）
    daily_pnl_amount: float = 0.0    # 当日累计盈亏（元）
    consecutive_losses: int = 0      # 连续亏损笔数
    total_trade_value: float = 0.0   # 当日做 T 总仓位（元）
    signal_pause_until: str = ""     # 信号暂停到何时（HH:MM 格式）
    same_direction_failures: int = 0 # 同向信号连续未成交次数


@dataclass
class AccountState:
    """全账户状态"""
    initial_capital: float = 1_000_000
    daily_pnl_pct: float = 0.0
    daily_pnl_amount: float = 0.0
    total_trade_value: float = 0.0   # 全账户做 T 总仓位


# ============================================================
# 仓位比例计算
# ============================================================

def get_position_ratio(market_type: MarketType,
                       gap_scenario: int = 0,
                       is_weak_market: bool = False) -> float:
    """
    根据行情状态计算做 T 仓位比例

    仓位基数 = 个股原有底仓

    Args:
        market_type: 行情分型
        gap_scenario: 跳空级别 (0=无, 1=小, 2=中, 3=大)
        is_weak_market: 大盘是否偏弱

    Returns:
        做 T 仓位 / 底仓 的比例
    """
    # 跳空优先
    if gap_scenario == 3:   # 跳空 > 2.3%
        return POSITION["gap_extreme"]
    elif gap_scenario == 2: # 跳空 1.5%-2.3%
        return POSITION["gap_large"]
    elif gap_scenario == 1: # 跳空 0.5%-1.5%
        return POSITION["gap_medium"]

    # 大盘偏弱
    if is_weak_market:
        return POSITION["weak_choppy"]

    # 按行情分型
    if market_type == MarketType.BULL or market_type == MarketType.BEAR:
        return POSITION["trend_confident"]
    elif market_type == MarketType.REVERSAL:
        return POSITION["weak_choppy"]  # 变盘期轻仓
    else:
        return POSITION["range_normal"]


# ============================================================
# PositionManager
# ============================================================

class PositionManager:
    """单标的仓位管理器"""

    def __init__(self, code: str, base_shares: int):
        self.state = PositionState(code=code, base_shares=base_shares)
        self.code = code

    def allocate_trade_size(self,
                            market_type: MarketType,
                            gap_scenario: int = 0,
                            is_weak_market: bool = False,
                            account_state: Optional[AccountState] = None) -> int:
        """
        计算本次交易的仓位（股数）

        Args:
            market_type: 行情分型
            gap_scenario: 跳空级别
            is_weak_market: 大盘偏弱
            account_state: 全账户状态

        Returns:
            可交易的股数（整手），0 表示不可交易
        """
        # 检查单标的累计亏损
        if self.state.daily_pnl_pct <= POSITION["single_stock_loss_halt_pct"]:
            return 0

        # 检查全账户止损触发
        if account_state is not None:
            # 全账户触发止损 → 降为 5%
            threshold = get_account_stop_threshold(account_state.daily_pnl_pct)
            if account_state.daily_pnl_pct <= -threshold:
                ratio = POSITION["account_loss_reduced_pct"]
                return int(self.state.base_shares * ratio / 100) * 100

        ratio = get_position_ratio(market_type, gap_scenario, is_weak_market)
        shares = int(self.state.base_shares * ratio / 100) * 100  # 整手
        return max(shares, 0)

    def check_position_limits(self, direction: str,
                              signal_pause_until: str = "") -> Tuple[bool, str]:
        """
        检查是否可以开新仓

        限制：
        - 单日最多 3 轮
        - 同一方向已有持仓不得叠加
        - 信号暂停期间不交易

        Args:
            direction: "long" / "short"
            signal_pause_until: 信号暂停到何时

        Returns:
            (can_open, reason)
        """
        if self.state.daily_rounds >= POSITION["max_rounds_per_stock"]:
            return (False, f"已达单日最大轮次 {self.state.max_rounds}")

        if signal_pause_until:
            return (False, f"信号暂停至 {signal_pause_until}")

        if direction == "long" and self.state.open_long_trade:
            return (False, "已有正T持仓，不叠加")
        if direction == "short" and self.state.open_short_trade:
            return (False, "已有反T持仓，不叠加")

        return (True, "")

    def register_open(self, direction: str, shares: int, price: float):
        """记录开仓"""
        if direction == "long":
            self.state.open_long_trade = True
        else:
            self.state.open_short_trade = True

        trade_value = shares * price
        self.state.total_trade_value += trade_value

    def register_close(self, direction: str, profit_pct: float, profit_amount: float):
        """记录平仓"""
        if direction == "long":
            self.state.open_long_trade = False
        else:
            self.state.open_short_trade = False

        self.state.daily_rounds += 1
        self.state.daily_pnl_pct += profit_pct
        self.state.daily_pnl_amount += profit_amount

        if profit_pct > 0:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

    def check_stock_halt(self) -> Tuple[bool, str]:
        """
        检查是否触发单标的停止交易

        Returns:
            (should_halt, reason)
        """
        if self.state.daily_pnl_pct <= POSITION["single_stock_loss_halt_pct"]:
            return (True, f"单标的累计亏损 {self.state.daily_pnl_pct:.2f}% >= {POSITION['single_stock_loss_halt_pct']}%")

        if self.state.consecutive_losses >= 2:
            return (True, f"连续亏损 {self.state.consecutive_losses} 笔")

        return (False, "")

    def get_state(self) -> PositionState:
        return self.state

    def reset_day(self):
        """重置当日状态（新交易日）"""
        self.state.daily_rounds = 0
        self.state.daily_pnl_pct = 0.0
        self.state.daily_pnl_amount = 0.0
        self.state.consecutive_losses = 0
        self.state.total_trade_value = 0.0
        self.state.open_long_trade = False
        self.state.open_short_trade = False
        self.state.signal_pause_until = ""
        self.state.same_direction_failures = 0


# ============================================================
# 全账户动态止损阈值
# ============================================================

def get_account_stop_threshold(daily_pnl_pct: float) -> float:
    """
    根据当日累计盈亏返回全账户止损阈值

    规则：
    - 累计盈利 >= 1.0% → 阈值 2.5%（有利润垫）
    - 盈亏在 -0.5%~+1.0% → 阈值 1.8%（标准）
    - 累计亏损已达 -1.5% → 阈值 1.5%（加速收紧）

    Args:
        daily_pnl_pct: 当日累计盈亏百分比

    Returns:
        止损阈值（正数）
    """
    cfg = ACCOUNT_STOP_LOSS

    if daily_pnl_pct >= cfg["profit_floor_high"] / 100:
        return cfg["threshold_profit"] / 100
    elif daily_pnl_pct >= cfg["profit_floor_neutral"] / 100:
        return cfg["threshold_neutral"] / 100
    else:
        return cfg["threshold_loss"] / 100


def check_account_stop(account: AccountState) -> Tuple[bool, str]:
    """
    检查全账户止损是否触发

    Returns:
        (should_stop, reason)
    """
    threshold = get_account_stop_threshold(account.daily_pnl_pct)
    loss_pct = abs(account.daily_pnl_pct) if account.daily_pnl_pct < 0 else 0

    if loss_pct >= threshold:
        return (True, f"全账户亏损 {account.daily_pnl_pct:.2f}% >= 阈值 {threshold:.2%}")

    return (False, "")


# ============================================================
# 全账户仓位红线检查
# ============================================================

def check_account_position_limits(account: AccountState,
                                  capital: float,
                                  new_trade_value: float) -> Tuple[bool, str]:
    """
    检查全账户仓位红线

    Args:
        account: 账户状态
        capital: 总资产
        new_trade_value: 新交易仓位金额

    Returns:
        (within_limits, reason)
    """
    total_after = account.total_trade_value + new_trade_value
    max_total = capital * POSITION["account_max_total"]

    if total_after > max_total:
        return (False, f"全账户做T仓位 {total_after:.0f} > {max_total:.0f} (20%)")

    return (True, "")
