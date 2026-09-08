#!/usr/bin/env python3
"""
宏观 + 微观多因子量化选股模型
月度再平衡入口

Usage:
  python main.py                        # 当月再平衡
  python main.py --month 2026-05        # 指定月份
  python main.py --manual-macro monetary=8,fiscal=6,meeting=8,credit=6,social=5
  python main.py --backtest 2025-06 2026-05   # 多期月度回测
"""
import argparse
import logging
from datetime import datetime

from dateutil.relativedelta import relativedelta

from config.settings import END_DATE, TEST_END_DATE
from core.rebalance import run_monthly_rebalance
from core.trader import simulate_holding_return, simulate_hk_holding_return

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_manual_macro(arg_str: str) -> dict:
    """解析手动宏观指标参数: 'monetary=8,fiscal=6' → {monetary_policy: 8, ...}"""
    result = {}
    for pair in arg_str.split(","):
        k, v = pair.strip().split("=")
        # 支持简写
        key_map = {
            "monetary": "monetary_policy",
            "fiscal": "fiscal_policy",
            "meeting": "meeting_tone",
            "credit": "credit_spread",
            "social": "social_financing",
        }
        key = key_map.get(k.strip(), k.strip())
        result[key] = int(v.strip())
    return result


def _iter_months(start_month: str, end_month: str) -> list:
    """生成 [start_month, end_month] 闭区间内的月份列表（"YYYY-MM"）"""
    cur = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")
    months = []
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=1)
    return months


def _latest_quarter_end(decision_date: datetime) -> str:
    """返回 decision_date 之前最近的季末（YYYYMMDD），作为财报截止日"""
    ends = [
        datetime(decision_date.year - 1, 12, 31),
        datetime(decision_date.year, 3, 31),
        datetime(decision_date.year, 6, 30),
        datetime(decision_date.year, 9, 30),
        datetime(decision_date.year, 12, 31),
    ]
    past = [d for d in ends if d < decision_date]
    return max(past).strftime("%Y%m%d")


def _measure_month(snapshot: dict, hold_start: str, hold_end: str) -> dict:
    """按某月快照测量持仓区间收益，返回分市场与合并结果"""
    target_codes = snapshot.get("portfolio", [])
    stock_weight = snapshot.get("stock_weight", 1.0)

    a_targets = [c for c in target_codes if len(str(c)) == 6]
    hk_targets = [c for c in target_codes if len(str(c)) == 5]

    a_net, _, _ = simulate_holding_return(
        a_targets, hold_start, hold_end, stock_weight=stock_weight,
    )
    hk_net = simulate_hk_holding_return(hk_targets, hold_start, hold_end) if hk_targets else 1.0

    n = len(target_codes) or 1
    combined = (a_net * len(a_targets) + hk_net * len(hk_targets)) / n

    return {
        "a_count": len(a_targets),
        "hk_count": len(hk_targets),
        "stock_weight": stock_weight,
        "a_net_value": round(a_net, 4),
        "hk_net_value": round(hk_net, 4),
        "combined_net_value": round(combined, 4),
        "monthly_return": round(combined - 1, 4),
    }


def run_multi_month_backtest(start_month: str, end_month: str, manual_macro: dict = None) -> list:
    """
    多期月度回测：逐月执行再平衡选股，测量该月持仓收益并汇总。

    每个月的口径：
      - 决策日 = 当月首日，用上月末前推的价格窗口与最近季末财报选股
      - 持仓区间 = 当月首日 → 下月首日
    """
    months = _iter_months(start_month, end_month)
    if len(months) < 2:
        print("多期回测需要至少两个不同的月份（如 --backtest 2025-06 2026-05）")
        return None

    print(f"\n{'='*60}")
    print(f"  多期月度再平衡回测  {start_month} → {end_month}（共 {len(months)} 期）")
    print(f"{'='*60}")

    records = []
    for m in months:
        m_dt = datetime.strptime(m, "%Y-%m")
        price_end = (m_dt - relativedelta(days=1)).strftime("%Y%m%d")
        price_start = (m_dt - relativedelta(months=3)).strftime("%Y%m%d")
        financial_date = _latest_quarter_end(m_dt)
        hold_start = m_dt.strftime("%Y%m%d")
        hold_end = (m_dt + relativedelta(months=1)).strftime("%Y%m%d")

        print(f"\n===== 期 {m}：选股（财报 {financial_date}，量价窗口 {price_start}~{price_end}）=====")
        snapshot = run_monthly_rebalance(
            target_month=m,
            manual_macro=manual_macro,
            financial_date=financial_date,
            price_start=price_start,
            price_end=price_end,
        )
        if snapshot is None:
            logger.error(f"{m}: 再平衡失败，跳过")
            records.append({"month": m, "error": "rebalance failed"})
            continue

        r = _measure_month(snapshot, hold_start, hold_end)
        r["month"] = m
        records.append(r)
        logger.info(
            f"{m}: A股{r['a_count']}只 港股{r['hk_count']}只 | "
            f"组合净值 {r['combined_net_value']:.4f} | 月收益 {r['monthly_return']:+.2%}"
        )

    ok = [r for r in records if "error" not in r]
    if not ok:
        print("无有效回测期")
        return records

    cum = 1.0
    for r in ok:
        cum *= r["combined_net_value"]
    months_n = len(ok)
    annual = cum ** (12.0 / months_n) - 1 if months_n > 0 else 0.0
    monthly_rets = [r["monthly_return"] for r in ok]
    avg_month = sum(monthly_rets) / len(monthly_rets)
    win_months = sum(1 for x in monthly_rets if x > 0)

    print(f"\n{'='*60}")
    print(f"  多期回测汇总（{months_n} 期有效）")
    print(f"{'='*60}")
    print(f"  累计净值:   {cum:.4f}")
    print(f"  累计收益:   {cum - 1:+.2%}")
    print(f"  年化收益:   {annual:+.2%}")
    print(f"  平均月收益: {avg_month:+.2%}")
    print(f"  正收益月数: {win_months}/{months_n}（胜率 {win_months / months_n:.1%}）")
    for r in ok:
        print(f"    {r['month']}: A股{r['a_count']} 港股{r['hk_count']} | "
              f"净值 {r['combined_net_value']:.4f} | 月收益 {r['monthly_return']:+.2%}")
    print(f"{'='*60}\n")

    return records


def main():
    parser = argparse.ArgumentParser(description="宏观+微观多因子量化选股")
    parser.add_argument("--month", type=str, default=None,
                        help="目标月份，默认当月（2026-05）")
    parser.add_argument("--manual-macro", type=str, default=None,
                        help="手动覆盖任意宏观指标评分(2-10)。简写: monetary/fiscal/meeting/credit/social "
                             "或直接用指标名如 social_financing=5")
    parser.add_argument("--backtest", nargs=2, metavar=("START", "END"),
                        help="多期回测，e.g. --backtest 2025-06 2026-05")
    args = parser.parse_args()

    manual_macro = None
    if args.manual_macro:
        manual_macro = parse_manual_macro(args.manual_macro)

    if args.backtest:
        run_multi_month_backtest(args.backtest[0], args.backtest[1], manual_macro=manual_macro)
        return

    # 月度再平衡
    target_month = args.month or "2026-05"
    snapshot = run_monthly_rebalance(
        target_month=target_month,
        manual_macro=manual_macro,
    )

    if snapshot is None:
        logger.error("月度再平衡失败！")
        return

    # 回测上期组合收益率
    target_codes = snapshot["portfolio"]
    stock_weight = snapshot["stock_weight"]

    print("\n========== 持仓回测（A股部分） ==========")
    a_targets = [c for c in target_codes if len(str(c)) == 6]
    net_value, profit, details = simulate_holding_return(
        a_targets, END_DATE, TEST_END_DATE, stock_weight=stock_weight,
    )
    logger.info(f"A股组合净值: {net_value:.4f} | 收益率: {profit:.2f}%")
    logger.info(f"股票仓位: {stock_weight:.1%} | 债券仓位: {details['bond_weight']:.1%}")

    hk_targets = [c for c in target_codes if len(str(c)) == 5]
    if hk_targets:
        hk_net = simulate_hk_holding_return(hk_targets, END_DATE, TEST_END_DATE)
        logger.info(
            f"港股入选: {len(hk_targets)} 只 | 组合净值: {hk_net:.4f} "
            f"| 收益率: {(hk_net - 1) * 100:.2f}%"
        )

    logger.info("====== 本次月度再平衡结束 ======")


if __name__ == "__main__":
    main()
