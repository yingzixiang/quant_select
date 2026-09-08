"""
intraday/main_intraday.py
日内 T+0 量化交易系统 CLI 入口

Usage:
    # 回测模式（自动筛选标的池）
    python -m intraday.main_intraday --mode backtest --start 2026-01-01 --end 2026-06-01

    # 回测模式（手动指定标的）
    python -m intraday.main_intraday --mode backtest --start 2026-01-01 --end 2026-06-01 \\
        --stocks 000858,600519,300750,002594

    # 指定参数矩阵
    python -m intraday.main_intraday --mode backtest --start 2026-05-01 --end 2026-06-01 \\
        --param-set high_active --capital 500000

    # 参数搜索模式
    python -m intraday.main_intraday --mode param-search --start 2026-01-01 --end 2026-03-01 \\
        --param atr_multiplier --range 1.2:1.8:0.1
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intraday.config import PARAMETER_MATRICES, PARAMETER_SEARCH_GRID


def mode_backtest(args):
    """回测模式"""
    from intraday.pipeline import run_backtest_range

    stock_codes = None
    if args.stocks:
        stock_codes = [s.strip() for s in args.stocks.split(",")]

    result = run_backtest_range(
        start_date=args.start,
        end_date=args.end,
        stock_codes=stock_codes,
        initial_capital=float(args.capital or 1_000_000),
        param_set=args.param_set or "normal",
        period=args.period or "5",
        base_position_shares=int(args.base_shares or 10000),
        pool_size=int(args.pool_size or 5),
    )

    if result.metrics:
        print(f"\n回测完成！交易 {result.metrics.get('total_trades', 0)} 笔")
        pass_count = sum(1 for v in result.acceptance.values() if v[0])
        print(f"验收通过: {pass_count}/{len(result.acceptance)}")


def mode_param_search(args):
    """参数敏感性搜索模式"""
    import numpy as np
    import pandas as pd
    from intraday.pipeline import run_backtest_range
    from datetime import datetime

    param_name = args.param
    range_str = args.range

    # 解析参数范围
    parts = range_str.split(":")
    if len(parts) != 3:
        print(f"参数范围格式错误: {range_str}，应为 start:end:step")
        return

    start_val, end_val, step_val = float(parts[0]), float(parts[1]), float(parts[2])

    # 获取基准参数
    base_param_set = args.param_set or "normal"
    base_params = PARAMETER_MATRICES[base_param_set].copy()

    # 找到要变化的参数在哪个矩阵键中
    grid_info = PARAMETER_SEARCH_GRID.get(param_name)
    if grid_info is None:
        print(f"未知参数: {param_name}，可用参数: {list(PARAMETER_SEARCH_GRID.keys())}")
        return

    # 参数到矩阵键的映射
    param_to_key = {
        "atr_multiplier": "atr_multiplier",
        "atr_window": "atr_window",
        "volume_ratio_threshold": "volume_ratio_threshold",
        "stop_loss_pct": "stop_loss_pct",
        "staggered_tp_ratio": "staggered_tp_ratio",
        "hysteresis_width": "hysteresis_width",
    }

    matrix_key = param_to_key.get(param_name, param_name)

    # 生成搜索值列表
    search_values = list(np.arange(start_val, end_val + step_val / 2, step_val))
    search_values = [round(v, 2) for v in search_values]

    print(f"\n{'='*60}")
    print(f"  参数敏感性搜索")
    print(f"  参数: {param_name} ({matrix_key})")
    print(f"  搜索范围: {search_values}")
    print(f"  回测区间: {args.start} ~ {args.end}")
    print(f"  基准参数: {base_param_set}")
    print(f"{'='*60}\n")

    results = []
    for val in search_values:
        print(f"\n--- 测试 {param_name} = {val} ---")

        # 修改参数矩阵（临时）
        import intraday.config as cfg
        original = cfg.PARAMETER_MATRICES[base_param_set].get(matrix_key, None)

        # 创建一个自定义参数集名称
        cfg.PARAMETER_MATRICES["_search_temp"] = base_params.copy()
        cfg.PARAMETER_MATRICES["_search_temp"][matrix_key] = val

        try:
            result = run_backtest_range(
                start_date=args.start,
                end_date=args.end,
                initial_capital=float(args.capital or 1_000_000),
                param_set="_search_temp",
                period=args.period or "5",
                pool_size=int(args.pool_size or 3),  # 参数搜索用更小的池
                verbose=False,
            )

            if result.metrics:
                results.append({
                    "param_value": val,
                    "win_rate": result.metrics.get("win_rate", 0),
                    "profit_loss_ratio": result.metrics.get("profit_loss_ratio", 0),
                    "total_return": result.metrics.get("total_return", 0),
                    "max_drawdown": result.metrics.get("max_drawdown", 0),
                    "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                    "total_trades": result.metrics.get("total_trades", 0),
                })

        finally:
            # 恢复
            if "_search_temp" in cfg.PARAMETER_MATRICES:
                del cfg.PARAMETER_MATRICES["_search_temp"]
            if original is not None:
                cfg.PARAMETER_MATRICES[base_param_set][matrix_key] = original

    # 输出结果
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print(f"  参数敏感性搜索结果: {param_name}")
        print(f"{'='*60}")
        print(df.to_string(index=False))

        # 保存 CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"output/param_search_{param_name}_{timestamp}.csv"
        os.makedirs("output", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存至 {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="A股底仓日内 T+0 量化交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["backtest", "param-search"],
                        help="运行模式")

    # 通用参数
    parser.add_argument("--start", type=str, default=None,
                        help="起始日期 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="结束日期")
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金（默认 100 万）")
    parser.add_argument("--param-set", type=str, default="normal",
                        choices=["narrow_range", "normal", "high_active"],
                        help="参数矩阵（默认 normal）")
    parser.add_argument("--period", type=str, default="5",
                        choices=["5", "1"],
                        help="K 线周期（默认 5min）")

    # 回测专用
    parser.add_argument("--stocks", type=str, default=None,
                        help="手动指定标的，逗号分隔（默认自动筛选）")
    parser.add_argument("--base-shares", type=int, default=10000,
                        help="每只标的底仓股数（默认 10000）")
    parser.add_argument("--pool-size", type=int, default=5,
                        help="标的池大小（默认 5）")

    # 参数搜索专用
    parser.add_argument("--param", type=str, default=None,
                        help="参数搜索目标参数名")
    parser.add_argument("--range", type=str, default=None,
                        help="参数搜索范围 start:end:step")

    args = parser.parse_args()

    if args.mode == "backtest":
        if not args.start or not args.end:
            print("回测模式需要 --start 和 --end 参数")
            sys.exit(1)
        mode_backtest(args)

    elif args.mode == "param-search":
        if not args.start or not args.end:
            print("参数搜索需要 --start 和 --end 参数")
            sys.exit(1)
        if not args.param or not args.range:
            print("参数搜索需要 --param 和 --range 参数")
            sys.exit(1)
        mode_param_search(args)


if __name__ == "__main__":
    main()
