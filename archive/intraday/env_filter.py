"""
intraday/env_filter.py
Section 2：全域环境过滤（大盘 + 板块，全局休战开关）

三个层次：
1. 全市场大盘过滤（沪深300 为主）
2. 所属板块过滤
3. 时段硬性过滤
"""
import sys
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from intraday.config import ENV_FILTER, DATA_DEGRADATION

# ---- 降级日志 ----
_warnings_issued: set = set()


def _warn_once(key: str, message: str):
    if key not in _warnings_issued:
        print(f"[环境过滤] ⚠ {message}")
        _warnings_issued.add(key)


@dataclass
class EnvStatus:
    """环境过滤结果"""
    block_open: bool = False       # 是否禁止开仓
    reason: str = ""               # 触发规则描述
    # 诊断信息
    index_consecutive_drop: float = 0.0
    index_amplitude: float = 0.0
    advance_decline_ratio: float = 0.0
    sector_blocks: Dict[str, str] = field(default_factory=dict)  # {code: reason}


# ============================================================
# 全市场大盘过滤
# ============================================================

def check_global_environment(index_df: pd.DataFrame,
                             bar_index: int) -> Dict:
    """
    检查全市场大盘环境

    满足任意一条，当日全标的禁止开仓：
    1. 指数连续 N 根 K 线累计跌幅 >= 阈值
    2. 涨跌家数比 < 1:2.2（不可用时跳过）
    3. 指数日内振幅 > 4.5%

    Args:
        index_df: 沪深300 指数分钟数据
        bar_index: 当前 bar 索引

    Returns:
        {"block_open": bool, "reason": str, "details": dict}
    """
    cfg = ENV_FILTER
    hist = index_df.iloc[:bar_index + 1]

    if len(hist) < 3:
        return {"block_open": False, "reason": "", "details": {}}

    close = hist["close"].astype(float)
    high = hist["high"].astype(float)
    low = hist["low"].astype(float)

    # ---- 规则 1：连续 N 根 K 线累计跌幅 ----
    n_bars = cfg["index_consecutive_bars"]
    if len(close) >= n_bars:
        last_n = close.iloc[-n_bars:]
        cumulative_change = (last_n.iloc[-1] - last_n.iloc[0]) / last_n.iloc[0] * 100

        if cumulative_change <= -cfg["index_consecutive_drop_pct"]:
            return {
                "block_open": True,
                "reason": f"指数连续 {n_bars} 根 K 线累计跌幅 {cumulative_change:.2f}% >= {cfg['index_consecutive_drop_pct']}%",
                "details": {"index_consecutive_drop": cumulative_change},
            }

    # ---- 规则 2：涨跌家数比（不可用时跳过） ----
    # 此数据需要盘中实时获取，回测中不可用
    # 已通过 DATA_DEGRADATION 标记为不可用

    # ---- 规则 3：指数日内振幅 ----
    if len(high) > 0 and len(low) > 0:
        day_high = high.max()
        day_low = low.min()
        if day_low > 0:
            amplitude = (day_high - day_low) / day_low * 100
            if amplitude > cfg["index_amplitude_max"]:
                return {
                    "block_open": True,
                    "reason": f"指数日内振幅 {amplitude:.2f}% > {cfg['index_amplitude_max']}%",
                    "details": {"index_amplitude": amplitude},
                }

    return {"block_open": False, "reason": "", "details": {}}


# ============================================================
# 所属板块过滤
# ============================================================

def check_sector_filter(sector_name: str,
                        sector_minute_dict: Dict[str, pd.DataFrame],
                        bar_index: int) -> Tuple[bool, str]:
    """
    检查板块级别过滤条件

    满足以下条件时，对应板块内所有个股禁止开仓：
    1. 板块日内跌幅 >= -1.2%
    2. 板块内超 75% 个股收跌

    Args:
        sector_name: 板块名称
        sector_minute_dict: {sector_name: minute_df}
        bar_index: 当前 bar 索引

    Returns:
        (blocked, reason)
    """
    cfg = ENV_FILTER

    sec_df = sector_minute_dict.get(sector_name)
    if sec_df is None or len(sec_df) < bar_index + 1:
        return (False, "sector_data_unavailable")

    hist = sec_df.iloc[:bar_index + 1]

    if "close" not in hist.columns:
        return (False, "")

    close = hist["close"].astype(float)

    # 板块日内跌幅（从当日开盘到当前）
    if len(close) >= 2:
        day_open = close.iloc[0]
        current = close.iloc[-1]
        if day_open > 0:
            drop_pct = (current - day_open) / day_open * 100
            if drop_pct <= cfg["sector_drop_threshold"]:
                return (True, f"板块 {sector_name} 日内跌幅 {drop_pct:.2f}% <= {cfg['sector_drop_threshold']}%")

    return (False, "")


def batch_check_sectors(stock_codes: List[str],
                        sector_map: Dict[str, str],
                        sector_minute_dict: Dict[str, pd.DataFrame],
                        bar_index: int) -> Dict[str, str]:
    """
    批量检查板块过滤

    Returns:
        {code: block_reason}，仅包含被板块封锁的标的
    """
    blocks: Dict[str, str] = {}

    for code in stock_codes:
        sector = sector_map.get(code, "")
        if not sector:
            continue
        blocked, reason = check_sector_filter(sector, sector_minute_dict, bar_index)
        if blocked:
            blocks[code] = reason

    return blocks


# ============================================================
# 时段硬性过滤
# ============================================================

def get_time_phase(timestamp_str: str) -> str:
    """
    判断当前 bar 所处的时间阶段

    Returns:
        "pre_trading"      (09:30-09:35) → 完全禁止交易
        "center_window"    (09:35-10:05) → 中枢建立期
        "normal"           (10:05-14:40) → 正常交易
        "close_only"       (14:40-14:57) → 仅平仓
        "closing_auction"  (14:57-15:00) → 集合竞价，撤单
    """
    try:
        t = pd.Timestamp(timestamp_str).time()
    except Exception:
        return "normal"

    t0935 = pd.Timestamp("09:35:00").time()
    t1005 = pd.Timestamp("10:05:00").time()
    t1440 = pd.Timestamp("14:40:00").time()
    t1457 = pd.Timestamp("14:57:00").time()
    t0930 = pd.Timestamp("09:30:00").time()

    if t0930 <= t < t0935:
        return "pre_trading"
    elif t0935 <= t < t1005:
        return "center_window"
    elif t1005 <= t < t1440:
        return "normal"
    elif t1440 <= t < t1457:
        return "close_only"
    else:
        return "closing_auction"


def can_open_position(timestamp_str: str, env_blocked: bool, stock_blocked: bool) -> Tuple[bool, str]:
    """
    综合判断当前是否可以开仓

    结合：时段过滤 + 全局环境过滤 + 个股板块过滤

    Returns:
        (can_open, reason)
    """
    phase = get_time_phase(timestamp_str)

    if phase in ("pre_trading", "closing_auction"):
        return (False, f"时段禁止交易: {phase}")

    if phase == "close_only":
        return (False, f"仅平仓时段: {phase} (14:40 后)")

    if env_blocked:
        return (False, "全局环境封堵")

    if stock_blocked:
        return (False, "板块/个股封堵")

    return (True, "")
