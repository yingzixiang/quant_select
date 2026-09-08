"""
概率校准 + 阈值重选（模型优化 A）

基于复盘数据（output/review/trades.csv）做两件事：
  1. 分箱 + isotonic 校准：把 XGBoost 输出的排序概率映射为「真实胜率」
  2. 阈值扫描：反推给定目标胜率下的最优 PREDICT_PROB_THRESHOLD

诚实声明：复盘样本有限（当前 ~119 笔），结论是「指示性」而非「确定性」。
建议积累更多复盘样本后重跑本脚本刷新阈值。

用法：
  python -m short_term.calibrate
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REVIEW_CSV = "output/review/trades.csv"
OUT_JSON = "output/review/calibration.json"


def load_trades(path=REVIEW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    # 修复前导零丢失：CSV 里 "001316" 可能被读成 "1316"
    df["code"] = df["code"].str.zfill(6)
    df = df[df["probability"].notna()]
    df = df[df["profit_pct"].notna()]
    df["is_win"] = df["profit_pct"] > 0
    return df


def bin_calibration(df: pd.DataFrame, n_bins: int = 8) -> list:
    """等频分箱，算每箱真实胜率与平均收益"""
    df = df.sort_values("probability")
    df = df.reset_index(drop=True)
    edges = np.linspace(0, len(df), n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        sub = df.iloc[lo:hi]
        bins.append({
            "prob_min": round(float(sub["probability"].min()), 4),
            "prob_max": round(float(sub["probability"].max()), 4),
            "n": int(len(sub)),
            "win_rate": round(float(sub["is_win"].mean()), 4),
            "avg_profit_pct": round(float(sub["profit_pct"].mean()), 4),
        })
    return bins


def fit_isotonic(df: pd.DataFrame):
    """isotonic 校准曲线：概率 → 真实胜率"""
    x = df["probability"].values
    y = df["is_win"].astype(float).values
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    y_hat = iso.fit_transform(x, y)
    # 记录校准曲线（在若干概率采样点上的映射）
    grid = np.linspace(0.0, 1.0, 21)
    curve = [{"prob": round(float(p), 2), "calibrated_win_rate": round(float(iso.predict([p])[0]), 4)}
             for p in grid]
    return curve


def threshold_scan(df: pd.DataFrame, target_win_rate: float, min_n: int = 10) -> dict:
    """扫描阈值：阈值以上样本的胜率/样本数/平均收益"""
    thresholds = np.arange(0.30, 0.90, 0.02)
    rows = []
    best = None
    for t in thresholds:
        sub = df[df["probability"] >= t]
        n = len(sub)
        if n == 0:
            continue
        wr = float(sub["is_win"].mean())
        avg = float(sub["profit_pct"].mean())
        rows.append({"threshold": round(float(t), 2), "n": int(n),
                     "win_rate": round(wr, 4), "avg_profit_pct": round(avg, 4)})
        # 选「胜率达标且样本最多」的最低阈值；若都不达标，取胜率最高者
        if wr >= target_win_rate and n >= min_n:
            if best is None or t < best["threshold"]:
                best = {"threshold": round(float(t), 2), "win_rate": round(wr, 4),
                        "n": int(n), "avg_profit_pct": round(avg, 4)}
    if best is None and rows:
        best_row = max(rows, key=lambda r: r["win_rate"])
        best = dict(best_row)
        best["note"] = f"无阈值达到目标胜率 {target_win_rate:.0%}，取胜率最高点"
    return {"scan": rows, "recommended": best}


def run(target_win_rate: float = 0.50, min_n: int = 10):
    df = load_trades()
    if len(df) < 30:
        print(f"[校准] 样本过少（{len(df)}），结果仅供参考")

    bins = bin_calibration(df)
    curve = fit_isotonic(df)
    scan = threshold_scan(df, target_win_rate, min_n)

    result = {
        "n_samples": int(len(df)),
        "overall_win_rate": round(float(df["is_win"].mean()), 4),
        "bins": bins,
        "isotonic_curve": curve,
        "threshold_scan": scan["scan"],
        "recommended_threshold": scan["recommended"],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 控制台报告
    print("=" * 60)
    print(f"  概率校准 + 阈值重选（{len(df)} 笔复盘样本）")
    print(f"  整体胜率: {result['overall_win_rate']:.1%}")
    print("=" * 60)
    print("\n[分箱校准] 概率区间 → 真实胜率")
    for b in bins:
        bar = "█" * int(b["win_rate"] * 20)
        print(f"  {b['prob_min']:.2f}~{b['prob_max']:.2f}  n={b['n']:3d}  "
              f"胜率 {b['win_rate']:6.1%}  {bar}")
    print("\n[阈值扫描] 阈值 → 阈值以上样本")
    for r in scan["scan"]:
        mark = " ← 推荐" if scan["recommended"] and abs(r["threshold"] - scan["recommended"]["threshold"]) < 1e-9 else ""
        print(f"  {r['threshold']:.2f}  n={r['n']:3d}  胜率 {r['win_rate']:6.1%}  "
              f"均收益 {r['avg_profit_pct']:+.2%}{mark}")
    rec = scan["recommended"]
    print(f"\n[推荐阈值] {rec['threshold']:.2f}（目标胜率 {target_win_rate:.0%}）"
          f"  胜率 {rec['win_rate']:.1%}  样本 {rec['n']}  均收益 {rec['avg_profit_pct']:+.2%}")
    if rec.get("note"):
        print(f"  注: {rec['note']}")
    print(f"\n[输出] {OUT_JSON}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="概率校准 + 阈值重选")
    parser.add_argument("--target-win", type=float, default=0.50, help="目标胜率（默认 0.50）")
    parser.add_argument("--min-n", type=int, default=10, help="推荐阈值最小样本数")
    args = parser.parse_args()
    run(target_win_rate=args.target_win, min_n=args.min_n)
