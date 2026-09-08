"""
短线量化选股系统 CLI 入口

Usage:
    python -m short_term.main_short --mode train    # 训练模型
    python -m short_term.main_short --mode select   # 今日选股
    python -m short_term.main_short --mode backtest --start 2025-01-01 --end 2026-05-01
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def mode_train(args):
    """训练模型（多日期采样，避免过拟合）"""
    from short_term.data_fetcher import fetch_a_stock_list, fetch_daily_kline_batch
    from short_term.alpha_factors import compute_alpha_factors_single
    from short_term.model_trainer import (
        prepare_labels, labels_to_binary, rolling_train, save_model,
        prepare_multi_date_dataset,
    )
    import pandas as pd
    import numpy as np

    train_end = args.date or (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
    target = datetime.strptime(train_end, "%Y%m%d")
    train_start = target - timedelta(days=504)  # 2年训练窗口
    print(f"[Train] 多日期训练: {train_start.strftime('%Y%m%d')} ~ {train_end}")

    # 获取股票列表（取成交活跃的Top N）
    stock_list = fetch_a_stock_list()
    n_stocks = min(args.sample or 800, len(stock_list))
    if "amount" in stock_list.columns:
        stock_list = stock_list.sort_values("amount", ascending=False)
    codes = stock_list.head(n_stocks)["code"].tolist()
    print(f"[Train] 股票池: {len(codes)} 只")

    # 获取日线（训练起始往前120天计算因子 + 训练结束往后14天计算标签）
    kline_start = train_start - timedelta(days=120)
    kline_end = target + timedelta(days=14)
    kline_dict = fetch_daily_kline_batch(
        codes,
        kline_start.strftime("%Y%m%d"),
        kline_end.strftime("%Y%m%d"),
    )
    print(f"[Train] 有效日线: {len(kline_dict)} 只")

    if len(kline_dict) < 100:
        print(f"[Train] 日线数据不足")
        return

    # 生成训练日期列表（每3个交易日采样一次）
    sample_code = list(kline_dict.keys())[0]
    sample_df = kline_dict[sample_code]
    all_dates = sorted(sample_df["date"].astype(str).str[:10].unique())
    train_dates = [d for d in all_dates
                   if train_start.strftime("%Y-%m-%d") <= d <= target.strftime("%Y-%m-%d")]
    # 每3天采样一次，跳过最后几天（需要未来数据算标签）
    sample_dates = [d.replace("-", "") for d in train_dates[::3][:-2]]
    print(f"[Train] 采样日期: {len(sample_dates)} 天（每3天一次）")

    # 多日期采样训练
    dataset = prepare_multi_date_dataset(kline_dict, sample_dates)
    if dataset is None:
        print("[Train] 多日期采样失败")
        return

    X_multi, y_multi, date_codes, feature_names = dataset

    # 训练
    result = rolling_train(
        X_multi=X_multi,
        y_multi=y_multi,
        feature_names_multi=feature_names,
    )

    if result:
        save_model(result)
        print(f"[Train] 训练完成，AUC: {result['metrics'].get('auc', 0):.4f}")
    else:
        print("[Train] 训练失败，样本不足")


def mode_select(args):
    """每日选股"""
    from short_term.daily_pipeline import run_daily_selection, save_selection_result

    target_date = args.date or datetime.now().strftime("%Y%m%d")
    result = run_daily_selection(target_date=target_date)

    if not result.empty:
        save_selection_result(result, target_date)
    else:
        print("[Select] 未选出符合条件的股票")


def mode_backtest(args):
    """回测模式"""
    from short_term.data_fetcher import fetch_a_stock_list, fetch_daily_kline_batch
    from short_term.backtest_engine import BacktestEngine, simple_select_func
    from datetime import datetime, timedelta

    start_date = args.start
    end_date = args.end

    print(f"[Backtest] 回测区间: {start_date} ~ {end_date}")

    # 获取股票列表
    stock_list = fetch_a_stock_list()
    codes = stock_list.sort_values("float_mv", ascending=False).head(1000)["code"].tolist()

    # 获取整个区间的日线数据
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    kline_start = start - timedelta(days=120)  # 留足计算窗口

    kline_dict = fetch_daily_kline_batch(
        codes,
        kline_start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
    )

    if len(kline_dict) < 100:
        print(f"[Backtest] 日线数据不足: {len(kline_dict)} 只")
        return

    # 生成交易日列表（用第一只股票的日期）
    sample_df = list(kline_dict.values())[0]
    date_col = None
    for col_name in ["date", "日期"]:
        if col_name in sample_df.columns:
            date_col = col_name
            break

    if date_col is None:
        print("[Backtest] 无法确定日期列，回测终止")
        return

    all_dates = sorted(sample_df[date_col].astype(str).str[:10].unique())
    selection_dates = [d for d in all_dates if start_date <= d <= end_date]

    if len(selection_dates) < 10:
        print(f"[Backtest] 交易日期不足: {len(selection_dates)}")
        return

    print(f"[Backtest] 交易日数: {len(selection_dates)}")

    # 执行回测
    engine = BacktestEngine(initial_capital=float(args.capital or 100_0000))

    engine.run(
        kline_dict=kline_dict,
        selection_dates=selection_dates,
        select_func=simple_select_func,
    )


def main():
    parser = argparse.ArgumentParser(description="短线量化选股系统")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "select", "backtest"],
                        help="运行模式: train(训练), select(选股), backtest(回测)")
    parser.add_argument("--date", type=str, default=None,
                        help="日期 YYYYMMDD（选股/训练模式）")
    parser.add_argument("--start", type=str, default=None,
                        help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=None,
                        help="回测初始资金（默认100万）")
    parser.add_argument("--sample", type=int, default=None,
                        help="股票样本数（训练/回测）")

    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "select":
        mode_select(args)
    elif args.mode == "backtest":
        if not args.start or not args.end:
            print("回测模式需要 --start 和 --end 参数")
            sys.exit(1)
        mode_backtest(args)


if __name__ == "__main__":
    main()
