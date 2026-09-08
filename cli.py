"""
统一策略 CLI（收敛重构 P3 新增入口，老入口 main.py / short_term.main_short 保留）

Usage:
    python cli.py --strategy monthly --month 2026-05
    python cli.py --strategy monthly --backtest 2025-06 2026-05
    python cli.py --strategy swing --date 20260903
"""
import argparse
import json
import os
import sys

from base.adapters import get_strategy


def _print_portfolio(p: "Portfolio"):
    print("=" * 60)
    print(f"  策略: {p.strategy}  日期: {p.date}")
    if p.stock_weight < 1.0:
        print(f"  股债配比: 股票 {p.stock_weight:.1%} / 债券 {p.bond_weight:.1%}")
    if p.style:
        print(f"  风格: {p.style}")
    print(f"  信号数: {len(p.signals)}")
    for s in p.signals:
        print(f"    [{s.market}] {s.code} {s.name:<8s} score={s.score:.4f}")
    if p.risk:
        print(f"  风控: {'通过' if p.risk.get('passed') else '存在违规'}")
        for v in p.risk.get("violations", []):
            print(f"    ⚠️ {v}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="量化选股统一策略入口")
    parser.add_argument("--strategy", type=str, required=True, choices=["monthly", "swing"],
                        help="策略名")
    # monthly 参数
    parser.add_argument("--month", type=str, default=None, help="中线目标月份 YYYY-MM")
    parser.add_argument("--manual-macro", type=str, default=None, help="手动宏观覆盖")
    parser.add_argument("--backtest", nargs=2, metavar=("START", "END"), help="中线多期回测")
    # swing 参数
    parser.add_argument("--date", type=str, default=None, help="短线选股日期 YYYYMMDD")
    args = parser.parse_args()

    strategy = get_strategy(args.strategy)

    if args.strategy == "monthly":
        if args.backtest:
            from main import run_multi_month_backtest
            run_multi_month_backtest(args.backtest[0], args.backtest[1])
            return
        p = strategy.run(target_month=args.month, manual_macro=None)
        _print_portfolio(p)
        _dump_portfolio(p, f"output/portfolio_unified_{p.date}.json")

    elif args.strategy == "swing":
        p = strategy.run(target_date=args.date)
        _print_portfolio(p)
        _dump_portfolio(p, f"output/selection_unified_{p.date}.json")


def _dump_portfolio(p: "Portfolio", path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(p.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n[CLI] 组合已保存: {path}")


if __name__ == "__main__":
    main()
