#!/usr/bin/env python3
"""
parameter_search.py
参数敏感性网格搜索

对 6 个关键参数进行网格搜索，输出 CSV 结果和敏感性报告。

Usage:
    python parameter_search.py --start 2026-01-01 --end 2026-03-01 --param atr_multiplier
    python parameter_search.py --start 2026-01-01 --end 2026-03-01 --all
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from intraday.config import PARAMETER_MATRICES, PARAMETER_SEARCH_GRID

# 参数名到矩阵键的映射
PARAM_TO_MATRIX_KEY = {
    "atr_multiplier": "atr_multiplier",
    "atr_window": "atr_window",
    "volume_ratio_threshold": "volume_ratio_threshold",
    "stop_loss_pct": "stop_loss_pct",
    "staggered_tp_ratio": "staggered_tp_ratio",
    "hysteresis_width": "hysteresis_width",
}


def run_single_param_search(param_name: str,
                            start_date: str,
                            end_date: str,
                            base_param_set: str = "normal",
                            capital: float = 1_000_000,
                            period: str = "5",
                            pool_size: int = 3,
                            stocks: list = None) -> pd.DataFrame:
    """
    对单个参数进行网格搜索

    Args:
        param_name: 参数名
        start_date: 起始日期
        end_date: 结束日期
        base_param_set: 基准参数矩阵
        capital: 初始资金
        period: K 线周期
        pool_size: 标的池大小（搜索时用小池节省时间）
        stocks: 手动指定标的

    Returns:
        DataFrame with search results
    """
    from intraday.pipeline import run_backtest_range

    grid_info = PARAMETER_SEARCH_GRID.get(param_name)
    if grid_info is None:
        print(f"未知参数: {param_name}")
        return pd.DataFrame()

    matrix_key = PARAM_TO_MATRIX_KEY.get(param_name, param_name)
    start_val, end_val, step_val = grid_info

    # 生成搜索值
    search_values = np.arange(start_val, end_val + step_val / 2, step_val)
    search_values = [round(float(v), 2) for v in search_values]

    print(f"\n{'='*60}")
    print(f"  参数敏感性网格搜索")
    print(f"  参数: {param_name} ({matrix_key})")
    print(f"  范围: {start_val} → {end_val} (步长 {step_val})")
    print(f"  测试点: {len(search_values)} 个: {search_values}")
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"{'='*60}\n")

    # 保存并修改参数矩阵
    import intraday.config as cfg
    base_params = cfg.PARAMETER_MATRICES[base_param_set].copy()
    original = base_params.get(matrix_key)

    results = []
    for i, val in enumerate(search_values):
        print(f"\n[{i+1}/{len(search_values)}] 测试 {param_name} = {val}")

        cfg.PARAMETER_MATRICES["_search_temp"] = base_params.copy()
        cfg.PARAMETER_MATRICES["_search_temp"][matrix_key] = val

        try:
            result = run_backtest_range(
                start_date=start_date,
                end_date=end_date,
                stock_codes=stocks,
                initial_capital=capital,
                param_set="_search_temp",
                period=period,
                pool_size=pool_size,
                verbose=False,
            )

            if result.metrics and result.metrics.get("total_trades", 0) > 0:
                m = result.metrics
                results.append({
                    "param_name": param_name,
                    "param_value": val,
                    "total_return": round(m.get("total_return", 0), 4),
                    "annual_return": round(m.get("annual_return", 0), 4),
                    "win_rate": round(m.get("win_rate", 0), 4),
                    "profit_loss_ratio": round(m.get("profit_loss_ratio", 0), 2),
                    "max_drawdown": round(m.get("max_drawdown", 0), 4),
                    "sharpe_ratio": round(m.get("sharpe_ratio", 0), 2),
                    "total_trades": m.get("total_trades", 0),
                    "false_signal_rate": round(m.get("false_signal_rate", 0), 4),
                })
            else:
                results.append({
                    "param_name": param_name,
                    "param_value": val,
                    "total_return": None,
                    "annual_return": None,
                    "win_rate": None,
                    "profit_loss_ratio": None,
                    "max_drawdown": None,
                    "sharpe_ratio": None,
                    "total_trades": 0,
                    "false_signal_rate": None,
                })

        finally:
            if "_search_temp" in cfg.PARAMETER_MATRICES:
                del cfg.PARAMETER_MATRICES["_search_temp"]

    # 恢复
    if original is not None:
        cfg.PARAMETER_MATRICES[base_param_set][matrix_key] = original

    return pd.DataFrame(results)


def run_all_params_search(start_date: str,
                          end_date: str,
                          capital: float = 1_000_000,
                          period: str = "5",
                          stocks: list = None):
    """对所有 6 个参数进行网格搜索"""
    all_results = []

    for param_name in PARAMETER_SEARCH_GRID.keys():
        df = run_single_param_search(
            param_name, start_date, end_date,
            capital=capital, period=period, stocks=stocks,
        )
        if not df.empty:
            all_results.append(df)

    if not all_results:
        print("无搜索结果")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"output/param_search_all_{timestamp}.csv"
    os.makedirs("output", exist_ok=True)
    combined.to_csv(csv_path, index=False)

    # 打印敏感性排名
    print(f"\n{'='*60}")
    print(f"  参数敏感性排名（按收益率标准差）")
    print(f"{'='*60}")

    sensitivity = combined.groupby("param_name")["total_return"].std().sort_values(ascending=False)
    for param, std in sensitivity.items():
        print(f"  {param:30s}  σ = {std:.4f}")

    print(f"\n  完整结果: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="参数敏感性网格搜索")
    parser.add_argument("--start", type=str, required=True, help="起始日期")
    parser.add_argument("--end", type=str, required=True, help="结束日期")
    parser.add_argument("--param", type=str, default=None,
                        help="单个参数名（不指定则搜索全部）")
    parser.add_argument("--all", action="store_true",
                        help="搜索全部参数")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="初始资金")
    parser.add_argument("--period", type=str, default="5",
                        help="K线周期")
    parser.add_argument("--stocks", type=str, default=None,
                        help="手动标的")
    parser.add_argument("--pool-size", type=int, default=3,
                        help="标的池大小")

    args = parser.parse_args()

    stocks_list = None
    if args.stocks:
        stocks_list = [s.strip() for s in args.stocks.split(",")]

    if args.all:
        run_all_params_search(args.start, args.end, args.capital,
                              args.period, stocks_list)
    elif args.param:
        df = run_single_param_search(args.param, args.start, args.end,
                                     capital=args.capital, period=args.period,
                                     pool_size=args.pool_size, stocks=stocks_list)
        if not df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"output/param_search_{args.param}_{timestamp}.csv"
            os.makedirs("output", exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"\n结果: {csv_path}")
            print(df.to_string(index=False))
    else:
        print("请指定 --param <参数名> 或 --all")
        print(f"可用参数: {list(PARAMETER_SEARCH_GRID.keys())}")


if __name__ == "__main__":
    main()
