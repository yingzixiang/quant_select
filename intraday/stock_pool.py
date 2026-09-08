"""
intraday/stock_pool.py
Section 1：标的池精选与动态淘汰

功能：
- select_stock_pool(): 回测前根据日线数据筛选 4-6 只标的
- check_dynamic_elimination(): 盘中逐 bar 检查淘汰条件
- detect_opening_gap(): 检测开盘跳空缺口并分级
"""
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

from intraday.config import STOCK_POOL, STOCK_POOL_DYNAMIC


# ============================================================
# 数据结构
# ============================================================

@dataclass
class StockPoolItem:
    """标的池中的一只标的"""
    code: str
    name: str = ""
    score: float = 0.0              # 综合评分（振幅 × 流动性）
    avg_amount_20d: float = 0.0     # 近 20 日日均成交额
    avg_turnover_20d: float = 0.0   # 近 20 日日均换手率
    avg_amplitude_5d: float = 0.0   # 近 5 日日均振幅
    blocked: bool = False           # 当日是否被封堵
    block_reason: str = ""
    gap_level: float = 0.0          # 今日跳空幅度（%）
    gap_scenario: int = 0           # 0=无, 1=小(0.5-1.5), 2=大(1.5-2.3), 3=极端(>2.3)


# ============================================================
# 准入筛选（回测前执行）
# ============================================================

def select_stock_pool(daily_kline_dict: Dict[str, pd.DataFrame],
                      stock_list_df: pd.DataFrame,
                      target_count: int = 5) -> List[StockPoolItem]:
    """
    从全 A 股中筛选符合日内 T+0 条件的标的池

    准入标准（全部满足）：
    1. 近 20 日日均成交额 >= 6 亿
    2. 日均换手率 3%-16%
    3. 近 5 日日内平均振幅 >= 3.2%
    4. 剔除 ST/*ST/退市/停牌 等风险标的
    5. 选取 4-6 只最优标的（振幅 × 流动性综合评分最高）

    Args:
        daily_kline_dict: {code: daily_kline_df}，每个 DataFrame 需含至少 20 行日线数据
        stock_list_df: A 股全量列表
        target_count: 目标标的数量（默认 5）

    Returns:
        List[StockPoolItem]，按评分降序排列
    """
    print(f"\n[标的池] ===== 精选标的池 =====")
    print(f"[标的池] 候选股票: {len(daily_kline_dict)} 只")
    sys.stdout.flush()

    candidates: List[StockPoolItem] = []

    for code, df in daily_kline_dict.items():
        if df is None or len(df) < 20:
            continue

        # ---- ST 过滤 ----
        name_match = stock_list_df[stock_list_df["code"] == code]
        if not name_match.empty:
            name = str(name_match.iloc[0].get("name", ""))
            if any(kw in name.upper() for kw in ["ST", "*ST", "退"]):
                continue

        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        if len(close) < 20:
            continue

        # ---- 日均成交额（20 日） ----
        if "amount_value" in df.columns:
            amount_series = df["amount_value"].astype(float).tail(20)
        else:
            amount_series = close.tail(20) * vol.tail(20) * 100
        avg_amount = amount_series.mean()
        if avg_amount < STOCK_POOL["min_avg_amount_20d"]:
            continue

        # ---- 日均换手率（20 日） ----
        turnover = 0.0
        if "turnover_rate" in df.columns:
            t_series = df["turnover_rate"].astype(float).tail(20)
            turnover = t_series.mean()
            if turnover < STOCK_POOL["min_turnover_rate"] or turnover > STOCK_POOL["max_turnover_rate"]:
                continue

        # ---- 近 5 日日内平均振幅 ----
        recent_5 = df.tail(5)
        amplitudes = []
        for _, row in recent_5.iterrows():
            amp = (row["high"] - row["low"]) / row["close"] * 100
            amplitudes.append(amp)
        avg_amplitude = np.mean(amplitudes) if amplitudes else 0
        if avg_amplitude < STOCK_POOL["min_avg_amplitude_5d"]:
            continue

        # ---- 综合评分（振幅 × 流动性） ----
        amount_score = min(avg_amount / 50_0000_0000, 2.0)  # 成交额归一化，上限 2.0
        amplitude_score = avg_amplitude / 5.0               # 振幅归一化
        score = amplitude_score * 0.6 + amount_score * 0.4   # 振幅权重更高

        item = StockPoolItem(
            code=code,
            name=str(name_match.iloc[0].get("name", "")) if not name_match.empty else "",
            score=score,
            avg_amount_20d=avg_amount,
            avg_turnover_20d=turnover,
            avg_amplitude_5d=avg_amplitude,
        )
        candidates.append(item)

    # ---- 排序取 Top N ----
    candidates.sort(key=lambda x: x.score, reverse=True)
    selected = candidates[:target_count]

    # 确保数量在 4-6 之间
    min_n = STOCK_POOL["target_pool_size_min"]
    max_n = STOCK_POOL["target_pool_size_max"]
    if len(selected) < min_n:
        selected = candidates[:min(min_n, len(candidates))]

    print(f"[标的池] 入选 {len(selected)} 只:")
    for item in selected:
        print(f"  {item.code} {item.name:8s}  "
              f"成交额:{item.avg_amount_20d/1e8:.1f}亿  "
              f"换手:{item.avg_turnover_20d:.1f}%  "
              f"振幅:{item.avg_amplitude_5d:.1f}%  "
              f"评分:{item.score:.3f}")
    print(f"[标的池] ===== 标的池筛选完成 =====\n")
    sys.stdout.flush()

    return selected


# ============================================================
# 开盘跳空检测
# ============================================================

def detect_opening_gap(minute_df: pd.DataFrame,
                       prev_day_close: float) -> Tuple[float, int]:
    """
    检测开盘跳空缺口并分级

    Args:
        minute_df: 当日分钟 K 线
        prev_day_close: 前一日收盘价

    Returns:
        (gap_pct, gap_scenario)
        gap_scenario: 0=无跳空, 1=小幅(0.5-1.5%), 2=中幅(1.5-2.3%), 3=大幅(>2.3%)
    """
    if minute_df is None or len(minute_df) < 1:
        return (0.0, 0)

    if prev_day_close <= 0:
        return (0.0, 0)

    # 第一根有交易的 K 线的开盘价或收盘价
    first_open = float(minute_df.iloc[0]["open"])
    gap_pct = (first_open - prev_day_close) / prev_day_close * 100

    rules = STOCK_POOL_DYNAMIC
    gap_abs = abs(gap_pct)

    if gap_abs < 0.5:
        scenario = 0
    elif gap_abs < 1.5:
        scenario = 1
    elif gap_abs < rules["gap_extreme_pct"]:
        scenario = 2
    else:
        scenario = 3

    return (gap_pct, scenario)


# ============================================================
# 盘中动态淘汰
# ============================================================

def check_dynamic_elimination(cur_price: float,
                              prev_close: float,
                              cur_turnover: Optional[float],
                              pool_items: List[StockPoolItem]) -> List[dict]:
    """
    盘中逐 bar 检查动态淘汰条件

    触发以下任一条件，移出当日标的池：
    1. 盘中触及 ±5% 涨跌幅
    2. 瞬时换手率 > 18%
    3. 开盘跳空高开/低开 > 2.3%（仓位降为 0，不移出池）

    Args:
        cur_price: 当前价格
        prev_close: 前日收盘价
        cur_turnover: 当前瞬时换手率（可能为 None）
        pool_items: 标的池

    Returns:
        List[dict]: 淘汰事件列表 [{code, reason, time}]
    """
    events = []
    rules = STOCK_POOL_DYNAMIC

    change_pct = (cur_price - prev_close) / prev_close * 100

    for item in pool_items:
        if item.blocked:
            continue

        # 涨跌幅触发
        if abs(change_pct) >= rules["price_limit_pct"]:
            item.blocked = True
            item.block_reason = f"涨跌幅触及 {change_pct:+.1f}%"
            events.append({"code": item.code, "reason": item.block_reason})

        # 瞬时换手率触发
        if cur_turnover is not None and cur_turnover > rules["turnover_spike"]:
            item.blocked = True
            item.block_reason = f"瞬时换手率 {cur_turnover:.1f}% > {rules['turnover_spike']}%"
            events.append({"code": item.code, "reason": item.block_reason})

    return events


# ============================================================
# 辅助：从标的池获取活跃代码列表
# ============================================================

def get_active_codes(pool_items: List[StockPoolItem]) -> List[str]:
    """返回未被封堵的标的代码列表"""
    return [item.code for item in pool_items if not item.blocked]


def get_pool_item(pool_items: List[StockPoolItem], code: str) -> Optional[StockPoolItem]:
    """按代码查找标的池项"""
    for item in pool_items:
        if item.code == code:
            return item
    return None
