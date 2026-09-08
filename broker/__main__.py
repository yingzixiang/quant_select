"""
下单对接 CLI

Usage:
    # 模拟盘：把某日选股结果转为买入委托并落盘（干跑，只打印不下单）
    python -m broker --file short_term/output/selection_20260903.json --capital 1000000 --dry-run

    # 模拟盘：真实下单（本地撮合，更新 output/paper_account.json）
    python -m broker --file short_term/output/selection_20260903.json --capital 1000000

    # 查询模拟盘账户
    python -m broker --query

    # 指定券商后端（当前仅 paper；真实渠道选定后注册进 broker/live.py）
    python -m broker --backend paper --file ... --capital ...
"""
import argparse
import json
import os
import sys

from broker.config import DEFAULT_CAPITAL, PAPER_STATE_FILE
from broker.bridge import load_selection, build_orders, akshare_price_provider
from broker.live import BrokerFactory


def _print_account(broker):
    print(f"可用资金: {broker.get_cash():,.2f}")
    print("当前持仓:")
    positions = broker.get_positions()
    if not positions:
        print("  （空）")
    for p in positions:
        print(f"  {p.code} {p.name:<8s} {p.quantity} 股  成本 {p.avg_price:.3f}")


def cmd_query():
    broker = BrokerFactory.create("paper")
    print("=" * 50)
    print("  模拟盘账户")
    print("=" * 50)
    _print_account(broker)
    print("=" * 50)


def cmd_trade(args):
    if not args.file:
        print("缺少 --file 选股结果路径")
        sys.exit(1)

    selection = load_selection(args.file)
    date = selection.get("date", "?")
    capital = args.capital or DEFAULT_CAPITAL

    broker = BrokerFactory.create(args.backend, capital=capital)
    orders, exit_rules, skipped = build_orders(selection, capital, akshare_price_provider)

    print("=" * 60)
    print(f"  信号下单桥  date={date}  backend={args.backend}  capital={capital:,.0f}")
    print("=" * 60)
    for s in skipped:
        print(f"  [跳过] {s}")

    if args.dry_run:
        print("\n[干跑] 以下委托不会实际执行：")
        for o in orders:
            print(f"  BUY {o.code} {o.name:<8s} {o.quantity} 股 @ {o.price}  ({o.reason})")
    else:
        print("\n[下单] 开始提交委托...")
        for o in orders:
            broker.place_order(o)
            print(f"  {o.status:<8s} {o.order_id}  {o.code} {o.name:<8s} "
                  f"{o.side} {o.quantity} 股 @ {o.price}  ({o.reason})")
        print("\n[账户]")
        _print_account(broker)

    print("\n[出场规则] 供盯盘程序消费：")
    for r in exit_rules:
        print(f"  {r['code']} {r['name']:<8s} 买入日 {r['buy_date']} → 卖出日 {r['sell_date']} | "
              f"止损 {r['stop_loss_pct']!s} | 止盈 {r['stop_profit_pct']!s}")

    # 落盘出场规则（无论干跑与否都写一份，供后续程序读取）
    exit_file = f"output/paper_exit_rules_{date}.json"
    os.makedirs("output", exist_ok=True)
    with open(exit_file, "w", encoding="utf-8") as f:
        json.dump({"date": date, "exit_rules": exit_rules}, f, ensure_ascii=False, indent=2)
    print(f"\n出场规则已保存: {exit_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="下单对接 CLI")
    parser.add_argument("--backend", type=str, default="paper", help="券商后端（当前仅 paper）")
    parser.add_argument("--file", type=str, default=None, help="选股结果 selection_*.json 路径")
    parser.add_argument("--capital", type=float, default=None, help="账户总资金")
    parser.add_argument("--dry-run", action="store_true", help="干跑：只打印，不下单")
    parser.add_argument("--query", action="store_true", help="查询模拟盘账户")
    args = parser.parse_args()

    if args.query:
        cmd_query()
    else:
        cmd_trade(args)


if __name__ == "__main__":
    main()
