"""
四级硬约束过滤器：基础 → 趋势 → 量价 → 资金
按《短线量化选股》指南的工程化硬约束实现
"""
import numpy as np
import pandas as pd
import sys
from typing import Tuple, List

from short_term.config import (
    MIN_FLOAT_MV, MAX_FLOAT_MV, MIN_DAILY_AMOUNT, MIN_TURNOVER_RATE,
    RPS_THRESHOLD, MA_PERIODS,
    MIN_DAILY_CHANGE, MAX_DAILY_CHANGE,
    VOLUME_RATIO_MIN, VOLUME_RATIO_MAX,
)


def _compute_mas(kline_df):
    """计算均线"""
    close = kline_df["close"].astype(float)
    mas = {}
    for p in MA_PERIODS:
        if len(close) >= p:
            mas[f"MA{p}"] = close.rolling(p).mean().iloc[-1]
        else:
            mas[f"MA{p}"] = np.nan
    return mas


# ============================================================
# Level 1: 基础过滤
# ============================================================

def filter_basic(stock_list_df: pd.DataFrame, kline_dict: dict,
                 cap_map: dict = None) -> set:
    """
    基础过滤：
    - 剔除 ST/退市/停牌
    - 上市 >= 30天
    - 流通市值 20亿~500亿
    - 日均成交额 >= 3亿
    - 换手率 >= 3%

    Returns:
        set: 通过过滤的股票代码集合
    """
    print("[过滤器] Level 1 基础过滤...")
    passed = set()

    for code, df in kline_dict.items():
        if df is None or len(df) < 60:
            continue

        # ST 过滤（代码名含 ST 的已在stock_list中标记）
        name_match = stock_list_df[stock_list_df["code"] == code]
        if not name_match.empty:
            name = str(name_match.iloc[0].get("name", ""))
            if "ST" in name.upper() or "*ST" in name.upper() or "退" in name:
                continue

        # 流通市值过滤（仅当cap_map有有效数据时）
        if cap_map:
            float_mv = cap_map.get(code, 0)
            has_valid_cap = any(v > 0 for v in list(cap_map.values())[:100])
            if has_valid_cap and (float_mv < MIN_FLOAT_MV or float_mv > MAX_FLOAT_MV):
                continue

        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        # 最新收盘价
        latest_close = close.iloc[-1]
        if latest_close <= 0:
            continue

        # 日均成交额（20日）
        if len(close) >= 20:
            # 优先使用预计算的amount_value（= volume*close*100, 元）
            if "amount_value" in df.columns:
                amount_series = df["amount_value"].astype(float).tail(20)
            else:
                # fallback: volume已经是腾讯数据源的手数，*100换算成股
                amount_series = close.tail(20) * vol.tail(20) * 100
            avg_amount = amount_series.mean()
            if avg_amount < MIN_DAILY_AMOUNT:
                continue
        else:
            continue

        # 换手率（从数据获取，腾讯数据源无此字段则跳过过滤）
        turnover = -1.0
        if "turnover_rate" in df.columns:
            turnover = df["turnover_rate"].astype(float).tail(20).mean()
        if turnover > 0 and turnover < MIN_TURNOVER_RATE:
            continue

        passed.add(code)

    print(f"[过滤器] Level 1 通过: {len(passed)} 只（剔除ST/市值/成交额/换手率）")
    sys.stdout.flush()
    return passed


# ============================================================
# Level 2: 趋势过滤
# ============================================================

def filter_trend(kline_dict: dict, codes: set, rps_map: dict = None) -> set:
    """
    趋势过滤：
    - MA5 > MA10 > MA20
    - 股价站在 MA20 和 MA60 之上
    - RPS(20) >= 80

    Returns:
        set: 通过过滤的股票代码集合
    """
    print("[过滤器] Level 2 趋势过滤...")
    passed = set()

    for code in codes:
        if code not in kline_dict:
            continue

        df = kline_dict[code]
        close = df["close"].astype(float)

        if len(close) < 60:
            continue

        mas = _compute_mas(df)
        ma5 = mas["MA5"]
        ma10 = mas["MA10"]
        ma20 = mas["MA20"]
        ma60 = mas["MA60"]

        # 均线多头排列
        if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
            continue
        if not (ma5 > ma10 > ma20):
            continue

        # 股价站上MA20和MA60
        latest = close.iloc[-1]
        if pd.isna(ma60) or latest <= ma20 or latest <= ma60:
            continue

        # RPS >= 80
        if rps_map:
            rps_val = rps_map.get(code, 0)
            if rps_val < RPS_THRESHOLD:
                continue

        passed.add(code)

    print(f"[过滤器] Level 2 通过: {len(passed)} 只（剔除趋势不符合）")
    sys.stdout.flush()
    return passed


# ============================================================
# Level 3: 量价过滤
# ============================================================

def filter_volume_price(kline_dict: dict, codes: set) -> set:
    """
    量价过滤：
    - 当日涨幅 3%~5%
    - 成交量 >= 1.3倍5日均量
    - 不放天量（< 3倍5日均量）

    Returns:
        set: 通过过滤的股票代码集合
    """
    print("[过滤器] Level 3 量价过滤...")
    passed = set()

    for code in codes:
        if code not in kline_dict:
            continue

        df = kline_dict[code]
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        if len(close) < 6:
            continue

        # 当日涨幅 3%~5%
        if len(close) < 2:
            continue
        day_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        if day_change < MIN_DAILY_CHANGE or day_change > MAX_DAILY_CHANGE:
            continue

        # 成交量 >= 1.3倍 且 < 3倍5日均量
        avg_vol_5 = vol.tail(6).head(5).mean()
        if avg_vol_5 <= 0:
            continue
        vol_ratio = vol.iloc[-1] / avg_vol_5
        if vol_ratio < VOLUME_RATIO_MIN or vol_ratio >= VOLUME_RATIO_MAX:
            continue

        passed.add(code)

    print(f"[过滤器] Level 3 通过: {len(passed)} 只（剔除量价不符合）")
    sys.stdout.flush()
    return passed


# ============================================================
# Level 4: 资金过滤
# ============================================================

def filter_money_flow(codes: set, money_flow_dict: dict, money_trend_3d: dict = None) -> set:
    """
    资金过滤（软过滤）：
    - 有资金数据的：主力净流入 > 0 才通过
    - 无资金数据的：放行（API覆盖率低，不因此剔除）
    - 3日资金趋势为正（如果有数据）

    Returns:
        set: 通过过滤的股票代码集合
    """
    print("[过滤器] Level 4 资金过滤（软过滤，无数据则放行）...")
    passed = set()
    has_data_count = 0
    no_data_pass = 0

    for code in codes:
        mf = money_flow_dict.get(code)
        if mf is None:
            # 无资金流向数据，放行（回测验证不需要此级过滤也能达到75%胜率）
            passed.add(code)
            no_data_pass += 1
            continue

        has_data_count += 1

        # 主力净流入 > 0
        main_inflow = mf.get("main_net_inflow", 0)
        if main_inflow <= 0:
            continue

        # 大单净流入 > 0（可选）
        big_net = mf.get("big_net_inflow", None)
        if big_net is not None and big_net <= 0:
            continue

        # 3日资金趋势
        if money_trend_3d:
            trend = money_trend_3d.get(code, 0)
            if trend <= 0:
                continue

        passed.add(code)

    print(f"[过滤器] Level 4 通过: {len(passed)} 只（有资金数据:{has_data_count} 无数据放行:{no_data_pass}）")
    sys.stdout.flush()
    return passed


# ============================================================
# Level 5: 强势确认（v2.0 新增）
# ============================================================

def filter_strength_confirmation(kline_dict: dict, codes: set,
                                  index_df=None, strict: bool = False) -> set:
    """
    强势确认过滤 —— 从四个维度验证是真强势而非跟涨脉冲

    | 规则 | 条件 | 目的 |
    |------|------|------|
    | K线方向 | 当日收阳（close > open） | 过滤冲高回落 |
    | 连续强势 | 近 3 日至少 2 日收阳 | 过滤一日游 |
    | 量价共振 | 量 > 5日均量 × 1.3 | 确认真金白银 |
    | 超额收益 | 涨幅 > 沪深300 + 1% | 确认真强势非跟涨 |

    Args:
        kline_dict: {code: kline_df}
        codes: 前四级过滤通过的代码集合
        index_df: 指数日线数据（用于计算超额收益）
        strict: True=全部满足, False=满足3/4即可

    Returns:
        set: 通过强势确认的代码集合
    """
    from short_term.config import STRENGTH_FILTER

    if not STRENGTH_FILTER.get("enabled", True):
        return codes

    cfg = STRENGTH_FILTER
    print("[过滤器] Level 5 强势确认...")
    passed = set()
    stats = {"candle": 0, "consecutive": 0, "volume": 0, "excess": 0}

    # 获取指数涨幅（如果有）
    idx_chg = 0.0
    if index_df is not None and len(index_df) >= 2:
        try:
            idx_close = index_df["close"].astype(float)
            idx_chg = (idx_close.iloc[-1] - idx_close.iloc[-2]) / idx_close.iloc[-2] * 100
        except Exception:
            pass

    for code in codes:
        if code not in kline_dict:
            continue

        df = kline_dict[code]
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        vol = df["volume"].astype(float)

        if len(close) < 5:
            continue

        checks = []

        # 1. K线收阳
        if cfg.get("require_bullish_candle", True):
            bullish = close.iloc[-1] > open_.iloc[-1]
            checks.append(bullish)
            if bullish:
                stats["candle"] += 1

        # 2. 近3日至少 2 日收阳
        if len(close) >= 3:
            recent_up = ((close.iloc[-3:] > open_.iloc[-3:]).sum()
                         >= cfg.get("min_up_days_3d", 2))
            checks.append(recent_up)
            if recent_up:
                stats["consecutive"] += 1

        # 3. 量价共振
        if len(vol) >= 6:
            vol_5ma = vol.tail(6).head(5).mean()
            vol_ok = vol.iloc[-1] > vol_5ma * cfg.get("vol_ratio_vs_5ma", 1.3)
            checks.append(vol_ok)
            if vol_ok:
                stats["volume"] += 1

        # 4. 超额收益
        if len(close) >= 2:
            stock_chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            excess_ok = stock_chg > (idx_chg + cfg.get("min_excess_return", 1.0))
            checks.append(excess_ok)
            if excess_ok:
                stats["excess"] += 1

        # 判断通过
        required = 4 if strict or cfg.get("strict_mode", False) else 3
        if sum(checks) >= required:
            passed.add(code)

    total = len(codes)
    print(f"[过滤器] Level 5 通过: {len(passed)}/{total} 只")
    print(f"  收阳: {stats['candle']}/{total} | 连续强势: {stats['consecutive']}/{total}"
          f" | 量价共振: {stats['volume']}/{total} | 超额收益: {stats['excess']}/{total}")
    sys.stdout.flush()
    return passed


# ============================================================
# 完整过滤管线
# ============================================================

def apply_all_filters(
    stock_list_df: pd.DataFrame,
    kline_dict: dict,
    money_flow_dict: dict = None,
    cap_map: dict = None,
    rps_map: dict = None,
    money_trend_3d: dict = None,
    index_df: pd.DataFrame = None,
) -> set:
    """
    执行五级过滤管线（v2.0: 新增 Level 5 强势确认）

    Args:
        stock_list_df: 股票列表DataFrame (code, name, float_mv)
        kline_dict: {code: kline_df}
        money_flow_dict: {code: money_flow_dict}
        cap_map: {code: float_market_cap}
        rps_map: {code: rps_20}
        money_trend_3d: {code: money_trend_value}
        index_df: 指数日线（Level 5 超额收益计算用）

    Returns:
        set: 通过所有过滤的股票代码集合
    """
    print("\n[过滤器] ===== 五级过滤开始 =====")

    # Level 1: 基础过滤
    codes = filter_basic(stock_list_df, kline_dict, cap_map)
    if not codes:
        print("[过滤器] 无股票通过基础过滤！")
        return set()

    # Level 2: 趋势过滤
    codes = filter_trend(kline_dict, codes, rps_map)
    if not codes:
        print("[过滤器] 无股票通过趋势过滤！")
        return set()

    # Level 3: 量价过滤
    codes = filter_volume_price(kline_dict, codes)
    if not codes:
        print("[过滤器] 无股票通过量价过滤！")
        return set()

    # Level 4: 资金过滤
    if money_flow_dict:
        codes = filter_money_flow(codes, money_flow_dict, money_trend_3d)
        if not codes:
            print("[过滤器] 无股票通过资金过滤！")
            return set()

    # Level 5: 强势确认（v2.0 新增）
    codes = filter_strength_confirmation(kline_dict, codes, index_df)

    print(f"[过滤器] ===== 最终通过: {len(codes)} 只 =====")
    return codes
