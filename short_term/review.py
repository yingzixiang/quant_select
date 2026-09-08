"""
短线选股复盘器：把每日选股信号结算成真实交易记录，沉淀复盘数据库供回测/优化

口径（v1，务实近似，可在 config 调整）：
  - 买入：buy_date 开盘价（selection 里 buy_time 为 09:30 开盘买入）
  - 卖出：sell_date 收盘价（15:00 收盘卖出）
  - 止损/止盈：持仓期内用日线 high/low 判定
      - 止损价 = 买入价 × (1 + stop_loss_pct)，止盈价 = 买入价 × (1 + stop_profit_pct)
      - 逐日：先判 high 触及止盈、low 触及止损；同一天两者都触发时保守取止损（风控优先）
      - 谁先触发谁先退出（日线无法精确到分钟，这是近似）
  - 数据：akshare 前复权日线（qfq），复用 core/data_cache 缓存

输出：
  - output/review/trades.csv   每笔明细（可被 pandas/Excel 直接分析）
  - output/review/summary.json 汇总指标（复用 base/backtest 统一指标层）

用法：
  python -m short_term.review                # 复盘全部历史 selection
  python -m short_term.review --date 20260903  # 复盘单日
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import akshare as ak

from core.data_cache import cached_api_call


REVIEW_DIR = "output/review"


def _pct(text):
    """从 '-5% 严格止损' / '+10% 可止盈' 解析出小数比例"""
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", str(text or ""))
    return float(m.group(1)) / 100.0 if m else None


def _as_date(s):
    """'2026-09-04' → datetime.date"""
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()


def fetch_kline(code: str, start: str, end: str):
    """拉取日线（前复权），返回按日期升序、含 open/high/low/close 的 DataFrame

    多源回退：东财 stock_zh_a_hist → 腾讯 stock_zh_a_hist_tx → 新浪 stock_zh_a_daily
    """
    prefix = "sh" if code.startswith(("6", "9")) else "sz"

    # 源1：东方财富（qfq 前复权）
    try:
        df = cached_api_call(
            ak.stock_zh_a_hist,
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="qfq",
            max_age_seconds=21600,
        )
        if df is not None and not df.empty and "开盘" in df.columns:
            return _normalize_kline(df)
    except Exception:
        pass

    # 源2：腾讯（未复权，但持仓期短、影响小）
    try:
        df = cached_api_call(
            ak.stock_zh_a_hist_tx,
            symbol=f"{prefix}{code}",
            start_date=start, end_date=end, adjust="",
            max_age_seconds=21600,
        )
        if df is not None and not df.empty and "open" in df.columns:
            return _normalize_kline(df)
    except Exception:
        pass

    # 源3：新浪（全历史，取区间）
    try:
        df = cached_api_call(
            ak.stock_zh_a_daily,
            symbol=f"{prefix}{code}", adjust="",
            max_age_seconds=21600,
        )
        if df is not None and not df.empty:
            df = _normalize_kline(df)
            s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
            return df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
    except Exception:
        pass

    return None


def _normalize_kline(df) -> pd.DataFrame:
    """把各源列名归一化到 date/open/high/low/close"""
    rename = {}
    for src, dst in [("日期", "date"), ("开盘", "open"), ("收盘", "close"),
                     ("最高", "high"), ("最低", "low")]:
        if src in df.columns:
            rename[src] = dst
    df = df.rename(columns=rename)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def settle_signal(sig: dict, buy_date: str, sell_date: str) -> dict:
    """
    结算单条选股信号，返回交易明细 dict

    退出原因：expired(到期) / stop_loss(止损) / stop_profit(止盈) / pending(数据不足)
    """
    code = sig["code"]
    name = sig.get("name", "")
    sl_pct = _pct(sig.get("stop_loss"))
    tp_pct = _pct(sig.get("stop_profit"))

    start = _as_date(buy_date)
    end = _as_date(sell_date)

    df = fetch_kline(code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    if df is None or len(df) < 2:
        return {"code": code, "name": name, "status": "pending", "reason": "no_kline"}

    # 买入 = buy_date 开盘价
    buy_rows = df[df["date"] == start]
    if buy_rows.empty:
        buy_rows = df.iloc[[0]]
    entry = float(buy_rows.iloc[0]["open"])
    if entry <= 0:
        return {"code": code, "name": name, "status": "pending", "reason": "bad_price"}

    stop_price = entry * (1 + sl_pct) if sl_pct is not None else None
    take_price = entry * (1 + tp_pct) if tp_pct is not None else None

    # 买入日后逐日判定（跳过买入日当天，因为开盘即买入）
    hold = df[df["date"] > start]
    exit_price = None
    exit_reason = "expired"
    exit_date = None

    for _, row in hold.iterrows():
        d = row["date"]
        high = float(row["high"])
        low = float(row["low"])
        hit_tp = take_price is not None and high >= take_price
        hit_sl = stop_price is not None and low <= stop_price
        if hit_sl and hit_tp:
            # 同一天两者都触发：保守取止损（风控优先）
            exit_price, exit_reason, exit_date = stop_price, "stop_loss", d
            break
        elif hit_sl:
            exit_price, exit_reason, exit_date = stop_price, "stop_loss", d
            break
        elif hit_tp:
            exit_price, exit_reason, exit_date = take_price, "stop_profit", d
            break

    if exit_reason == "expired":
        # 到期：sell_date 收盘价（若该日无数据，取最后一日收盘）
        sell_rows = df[df["date"] == end]
        last = sell_rows.iloc[0] if not sell_rows.empty else df.iloc[-1]
        exit_price = float(last["close"])
        exit_date = last["date"]

    profit_pct = (exit_price - entry) / entry
    hold_days = (exit_date - start).days if exit_date else (end - start).days

    return {
        "code": code,
        "name": name,
        "status": "closed",
        "buy_date": str(start),
        "sell_date": str(exit_date),
        "hold_days": int(hold_days),
        "entry": round(entry, 3),
        "exit": round(exit_price, 3),
        "profit_pct": round(profit_pct, 5),
        "is_win": bool(profit_pct > 0),
        "exit_reason": exit_reason,
        "probability": sig.get("probability"),
        "final_score": sig.get("final_score"),
        "position": sig.get("position"),
    }


def _iter_selections(date: str = None):
    """产出 (selection_date, sig) 迭代器"""
    pattern = f"short_term/output/selection_{date}.json" if date else "short_term/output/selection_*.json"
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sel_date = data.get("date", os.path.basename(path).replace("selection_", "").replace(".json", ""))
        for sig in data.get("selections", []):
            if sig.get("buy_date") and sig.get("sell_date"):
                yield sel_date, sig


def run_review(date: str = None) -> dict:
    """复盘全部/单日历史选股信号，写明细 CSV + 汇总 JSON"""
    os.makedirs(REVIEW_DIR, exist_ok=True)

    rows = []
    for sel_date, sig in _iter_selections(date):
        print(f"[复盘] 结算 {sel_date} {sig['code']} {sig.get('name','')} ...")
        sys.stdout.flush()
        try:
            t = settle_signal(sig, sig["buy_date"], sig["sell_date"])
            t["selection_date"] = sel_date
            rows.append(t)
        except Exception as e:
            rows.append({
                "code": sig.get("code"), "name": sig.get("name", ""),
                "status": "pending", "reason": f"error: {e}",
                "selection_date": sel_date,
            })

    # 写明细 CSV
    csv_path = os.path.join(REVIEW_DIR, "trades.csv")
    closed = [r for r in rows if r.get("status") == "closed"]
    fieldnames = ["selection_date", "code", "name", "buy_date", "sell_date", "hold_days",
                  "entry", "exit", "profit_pct", "is_win", "exit_reason",
                  "probability", "final_score", "position"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in closed:
            w.writerow(r)

    # 汇总
    summary = _summarize(closed, rows)
    summary_path = os.path.join(REVIEW_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[复盘] 明细已写: {csv_path}")
    print(f"[复盘] 汇总已写: {summary_path}")
    return summary


def _summarize(closed: list, all_rows: list) -> dict:
    """汇总指标：整体 + 按概率分层"""
    from base.backtest import compute_backtest_metrics

    n = len(closed)
    pending = len(all_rows) - n

    if n == 0:
        return {"closed": 0, "pending": pending, "message": "无已结算交易"}

    profits = np.array([r["profit_pct"] for r in closed], dtype=float)

    # 用统一指标层算整体指标（此处 equity 用每笔收益近似构造，仅算胜率/盈亏比，其余用简化口径）
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0

    # 按买入日构造净值曲线（等权、单票 20% 仓位，简化）
    df = pd.DataFrame(closed)
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df = df.sort_values("buy_date")
    daily_ret = df.groupby("buy_date")["profit_pct"].mean() * 0.20  # 每日等权 × 单票仓位
    equity = (1 + daily_ret).cumprod()

    metrics = compute_backtest_metrics(
        trades_pct=profits,
        equity=equity,
        initial_capital=1.0,
        final_value=float(equity.iloc[-1]) if len(equity) else 1.0,
    )

    # 按模型概率分层（验证概率是否单调有效）
    buckets = {}
    for r in closed:
        p = r.get("probability")
        if p is None:
            continue
        b = f"{int(p * 10) * 10}~{int(p * 10) * 10 + 10}%"
        buckets.setdefault(b, []).append(r["profit_pct"])

    prob_layers = []
    for b in sorted(buckets.keys()):
        ps = np.array(buckets[b])
        prob_layers.append({
            "layer": b,
            "n": int(len(ps)),
            "win_rate": round(float((ps > 0).sum() / len(ps)), 4),
            "avg_profit_pct": round(float(ps.mean()), 4),
        })

    exit_reasons = {}
    for r in closed:
        exit_reasons[r["exit_reason"]] = exit_reasons.get(r["exit_reason"], 0) + 1

    return {
        "closed": n,
        "pending": pending,
        "win_rate": round(float((profits > 0).sum() / n), 4),
        "profit_loss_ratio": round(avg_win / avg_loss, 4) if avg_loss > 0 else None,
        "avg_profit_pct": round(float(profits.mean()), 4),
        "total_return_equity": round(float(equity.iloc[-1] - 1) if len(equity) else 0.0, 4),
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "exit_reasons": exit_reasons,
        "prob_layers": prob_layers,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="短线选股复盘器")
    parser.add_argument("--date", type=str, default=None, help="只复盘某日 YYYYMMDD")
    args = parser.parse_args()
    run_review(date=args.date)
