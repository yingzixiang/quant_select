"""
intraday/pipeline.py
日内 T+0 回测编排器

串联全部十一层：
数据加载 → 标的筛选 → 回测执行 → 结果输出
"""
import sys
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from intraday.config import (
    BACKTEST_DEFAULTS, PARAMETER_MATRICES,
)
from intraday.data_fetcher import (
    fetch_a_stock_list,
    fetch_daily_kline_for_pool,
    preload_backtest_data,
)
from intraday.stock_pool import (
    select_stock_pool, StockPoolItem,
)
from intraday.backtest_engine import (
    IntradayBacktestEngine, BacktestResult,
)


def run_backtest_range(start_date: str,
                       end_date: str,
                       stock_codes: Optional[List[str]] = None,
                       initial_capital: float = 1_000_000,
                       param_set: str = "normal",
                       period: str = "5",
                       base_position_shares: int = 10000,
                       pool_size: int = 5,
                       verbose: bool = True) -> BacktestResult:
    """
    执行多日回测

    流程：
    1. 获取 A 股全量列表
    2. 获取日线数据 → 筛选标的池（4-6 只）
    3. 预加载分钟数据（标的池 + 指数 + 板块）
    4. 执行回测
    5. 输出结果

    Args:
        start_date: 起始日期 YYYYMMDD 或 YYYY-MM-DD
        end_date: 结束日期
        stock_codes: 手动指定标的池（跳过自动筛选）
        initial_capital: 初始资金
        param_set: 参数矩阵名称
        period: K 线周期
        base_position_shares: 每只标的底仓股数
        pool_size: 标的池大小
        verbose: 是否打印详细输出

    Returns:
        BacktestResult
    """
    # 统一日期格式
    if len(start_date) == 8:
        start_date_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    else:
        start_date_fmt = start_date
        start_date = start_date.replace("-", "")

    if len(end_date) == 8:
        end_date_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
    else:
        end_date_fmt = end_date
        end_date = end_date.replace("-", "")

    print(f"\n{'='*60}")
    print(f"  日内 T+0 量化回测")
    print(f"  区间: {start_date_fmt} ~ {end_date_fmt}")
    print(f"  K线周期: {period}min | 参数矩阵: {param_set}")
    print(f"  初始资金: {initial_capital:,.0f}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    # ---- Step 1: 标的池筛选 ----
    pool_items: List[StockPoolItem] = []
    base_positions: Dict[str, int] = {}

    if stock_codes:
        # 手动指定标的池
        print(f"[管线] 使用手动指定标的: {stock_codes}")
        for code in stock_codes:
            item = StockPoolItem(code=code)
            pool_items.append(item)
            base_positions[code] = base_position_shares
    else:
        # 自动筛选标的池
        print("[管线] Step 1/3: 筛选标的池...")
        sys.stdout.flush()

        # 日线数据窗口（回测前 20 天用于计算指标）
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        daily_start = (start_dt - timedelta(days=30)).strftime("%Y%m%d")

        stock_list = fetch_a_stock_list()
        if stock_list.empty:
            print("[管线] 错误: 无法获取股票列表")
            return BacktestResult()

        # 取 Top 200 成交活跃的股票做初筛
        if "amount" in stock_list.columns:
            stock_list = stock_list.sort_values("amount", ascending=False)
        top200 = stock_list.head(200)["code"].tolist()

        print(f"[管线] 获取 Top 200 日线数据 ({daily_start}~{end_date})...")
        daily_dict = fetch_daily_kline_for_pool(top200, daily_start, end_date)
        print(f"[管线] 有效日线: {len(daily_dict)} 只")

        if len(daily_dict) < 5:
            print("[管线] 日线数据不足")
            return BacktestResult()

        pool_items = select_stock_pool(daily_dict, stock_list, target_count=pool_size)

        for item in pool_items:
            base_positions[item.code] = base_position_shares

    if not pool_items:
        print("[管线] 无标的入选标的池！")
        return BacktestResult()

    valid_codes = [item.code for item in pool_items]

    # ---- Step 2: 预加载分钟数据 ----
    print("[管线] Step 2/3: 预加载分钟数据...")
    sys.stdout.flush()

    data = preload_backtest_data(valid_codes, start_date, end_date, period)

    if not data["stock_minute"]:
        print("[管线] 错误: 分钟数据加载失败")
        return BacktestResult()

    # ---- Step 3: 执行回测 ----
    print("[管线] Step 3/3: 执行回测...")
    sys.stdout.flush()

    engine = IntradayBacktestEngine(
        initial_capital=initial_capital,
        param_set_name=param_set,
        period=period,
    )

    result = engine.run(
        stock_minute_dict=data["stock_minute"],
        index_minute_df=data["index_minute"],
        sector_minute_dict=data["sector_minute"],
        sector_map=data["sector_map"],
        volume_profile=data["volume_profile"],
        pool_items=pool_items,
        base_positions=base_positions,
    )

    # ---- 保存结果 ----
    save_backtest_result(result, start_date, end_date, param_set)

    return result


def save_backtest_result(result: BacktestResult,
                         start_date: str,
                         end_date: str,
                         param_set: str):
    """保存回测结果到 JSON"""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"intraday_backtest_{start_date}_{end_date}_{param_set}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # 汇总输出
    output = {
        "start_date": start_date,
        "end_date": end_date,
        "param_set": param_set,
        "generated_at": datetime.now().isoformat(),
        "metrics": result.metrics,
        "acceptance": {k: {"passed": v[0], "detail": v[1]}
                       for k, v in result.acceptance.items()},
        "trade_count": len(result.trades),
        "trades": [
            {
                "code": t.stock_code,
                "direction": t.direction,
                "scenario": t.scenario,
                "open_time": t.open_time,
                "close_time": t.close_time,
                "profit_pct": round(t.profit_pct, 4),
                "is_win": t.is_win,
                "stop_reason": t.stop_reason,
            }
            for t in result.trades[:200]  # 最多保存 200 笔明细
        ],
        "daily_values": [
            {"date": v[0], "value": round(v[1], 2)}
            for v in result.daily_values
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[管线] 结果已保存至 {filepath}")
