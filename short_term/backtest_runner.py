"""
回测运行器：逐日模拟完整选股管线 + 买卖交易

用法:
    python -m short_term.backtest_runner --start 2026-01-01 --end 2026-05-01
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from short_term.config import (
    SINGLE_STOCK_WEIGHT, MAX_POSITION_COUNT, HOLD_DAYS,
    COMMISSION_RATE, SLIPPAGE, BACKTEST_BENCHMARKS,
)
from base.data_fetch import fetch_a_stock_list, fetch_daily_kline_batch
from short_term.alpha_factors import compute_alpha_factors_single
from short_term.model_trainer import load_model


def _safe_single_stock_factor(kline_df):
    """计算单只股票因子（仅alpha因子用于模型，另算过滤器所需指标）"""
    if kline_df is None or len(kline_df) < 60:
        return None

    try:
        close = kline_df["close"].astype(float)
        vol = kline_df["volume"].astype(float)
        open_ = kline_df["open"].astype(float)
        high = kline_df["high"].astype(float)
        low = kline_df["low"].astype(float)

        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20

        day_chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0
        vol_ratio = vol.iloc[-1] / vol.tail(6).head(5).mean() if vol.tail(6).head(5).mean() > 0 else 1
        avg_amount = (close.tail(20) * vol.tail(20)).mean() * 100
        turnover = 0.0
        if "turnover_rate" in kline_df.columns:
            t = kline_df["turnover_rate"].astype(float).tail(20).mean()
            if t > 0:
                turnover = t

        # Alpha因子
        alpha = compute_alpha_factors_single(kline_df) or {}

        # 补充自定义量价/K线因子（与训练时保持一致）
        hl_range = high.iloc[-1] - low.iloc[-1]
        alpha["candle_body_ratio"] = (close.iloc[-1] - open_.iloc[-1]) / hl_range if hl_range != 0 else 0
        body_high = max(open_.iloc[-1], close.iloc[-1])
        alpha["upper_shadow_ratio"] = (high.iloc[-1] - body_high) / hl_range if hl_range != 0 else 0
        body_low = min(open_.iloc[-1], close.iloc[-1])
        alpha["lower_shadow_ratio"] = (body_low - low.iloc[-1]) / hl_range if hl_range != 0 else 0
        alpha["open_gap"] = open_.iloc[-1] / close.iloc[-2] - 1 if len(close) >= 2 and close.iloc[-2] != 0 else 0
        alpha["intraday_volatility"] = hl_range / close.iloc[-1] if close.iloc[-1] != 0 else 0

        if len(close) >= 20:
            c5 = open_.tail(5).corr(vol.tail(5))
            c20 = open_.tail(20).corr(vol.tail(20))
            alpha["vol_price_corr_diff"] = c5 - c20 if not (np.isnan(c5) or np.isnan(c20)) else 0
        else:
            alpha["vol_price_corr_diff"] = 0

        if len(close) >= 2:
            vc = vol.iloc[-1] - vol.iloc[-2]
            pc = close.iloc[-1] - close.iloc[-2]
            alpha["vol_price_slope"] = vc / pc if abs(pc) >= 1e-6 else 0
        else:
            alpha["vol_price_slope"] = 0

        avg_v5 = vol.tail(5).mean()
        vr = vol.iloc[-1] / avg_v5 if avg_v5 > 0 else 1.0
        is_pb = close.iloc[-1] < close.iloc[-2] if len(close) >= 2 else False
        alpha["shrink_pullback"] = (0.7 - vr) / 0.7 if (is_pb and vr < 0.7) else 0

        m5v = vol.tail(6).head(5).mean()
        m20v = vol.tail(21).head(20).mean() if len(vol) >= 21 else m5v
        bl = max(m5v, m20v)
        alpha["volume_breakout_strength"] = min(vol.iloc[-1] / bl / 3.0, 1.0) if bl > 0 else 0

        ret3 = close.pct_change(3).iloc[-1] if len(close) >= 4 else 0
        vr3 = vol.tail(3).mean() / vol.tail(6).head(3).mean() if len(vol) >= 6 else 1
        alpha["ret_3d"] = ret3
        alpha["money_trend_3d"] = ret3 * vr3

        return {
            "close": close.iloc[-1],
            "day_change_pct": day_chg * 100,
            "vol_ratio": vol_ratio,
            "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
            "avg_amount_20": avg_amount,
            "turnover_rate": turnover,
            "alpha_factors": alpha,
        }
    except Exception:
        return None


def quick_filter_check(info):
    """快速硬约束过滤（放宽版：让模型做主要决策）"""
    if info is None:
        return False

    # Level 1: 基础（成交额 + 换手率）
    if info["avg_amount_20"] < 2_0000_0000:  # 日均成交额 >= 2亿（放宽）
        return False
    if info["turnover_rate"] < 1.0:  # 换手率 >= 1%（放宽）
        return False

    # Level 2: 趋势（均线多头 + 站上均线）
    if not (info["ma5"] > info["ma10"] > info["ma20"]):
        return False
    if info["close"] <= info["ma20"]:
        return False

    # Level 3: 量价（放宽涨幅区间）
    if info["day_change_pct"] < 1.0 or info["day_change_pct"] > 9.0:
        return False
    if info["vol_ratio"] < 1.1:  # 至少微幅放量（放宽）
        return False

    return True


def run_backtest(start_date: str, end_date: str, sample_size: int = 500,
                 prob_threshold: float = 0.0):
    """执行回测

    Args:
        prob_threshold: 模型概率最低阈值，低于此值的候选股直接跳过。
                        0.0 表示不过滤，0.55 表示只买模型比较看好的。
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    kline_start = start_dt - timedelta(days=120)
    kline_end = end_dt + timedelta(days=10)

    # 加载模型
    model_info = load_model()
    has_model = model_info is not None and "model" in model_info
    feature_cols = model_info.get("feature_cols", []) if has_model else []
    print(f"[回测] 模型: {'已加载' if has_model else '无'}  |  "
          f"特征数: {len(feature_cols)}  |  "
          f"AUC: {model_info.get('metrics', {}).get('auc', 'N/A') if has_model else 'N/A'}")

    # 获取股票列表
    print("[回测] 获取股票列表...")
    stock_list = fetch_a_stock_list()
    if sample_size > len(stock_list):
        sample_size = len(stock_list)
    top_codes = stock_list.head(sample_size)["code"].tolist()

    # 一次性拉取全部日线
    print(f"[回测] 获取 {len(top_codes)} 只股票日线 {kline_start.strftime('%Y%m%d')}~{kline_end.strftime('%Y%m%d')}...")
    kline_dict = fetch_daily_kline_batch(
        top_codes,
        kline_start.strftime("%Y%m%d"),
        kline_end.strftime("%Y%m%d"),
    )
    print(f"[回测] 有效日线: {len(kline_dict)} 只")

    # 生成交易日列表
    sample_code = list(kline_dict.keys())[0]
    sample_df = kline_dict[sample_code]
    date_col = "date"
    all_dates = sorted(sample_df[date_col].astype(str).str[:10].unique())
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]
    print(f"[回测] 交易日数: {len(trade_dates)}, {trade_dates[0]} ~ {trade_dates[-1]}")

    # ===== 回测模拟 =====
    initial_capital = 100_0000
    cash = initial_capital
    positions = []        # [(code, buy_date, buy_price, shares, cost)]
    closed_trades = []
    daily_values = {}

    filter_stats = {"total": 0, "l1_pass": 0, "l2_pass": 0, "l3_pass": 0,
                    "candidates": 0, "prob_filtered": 0, "max_prob": 0}

    for di, today in enumerate(trade_dates):
        # 1. 卖出到期的持仓
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        for pos in positions[:]:
            buy_dt = datetime.strptime(pos[1], "%Y-%m-%d")
            if (today_dt - buy_dt).days >= HOLD_DAYS:
                sell_df = kline_dict.get(pos[0])
                if sell_df is not None:
                    sell_row = sell_df[sell_df[date_col].astype(str).str[:10] == today]
                    sell_price = float(sell_row["close"].iloc[0]) if not sell_row.empty else pos[2]
                else:
                    sell_price = pos[2]

                sell_price_eff = sell_price * (1 - SLIPPAGE)
                revenue = pos[3] * sell_price_eff * (1 - COMMISSION_RATE)
                profit = revenue - pos[4]
                profit_pct = profit / pos[4] if pos[4] > 0 else 0
                cash += revenue
                closed_trades.append(profit_pct)
                positions.remove(pos)

        # 2. 选股
        if len(positions) < MAX_POSITION_COUNT:
            day_infos = {}
            for code in top_codes:
                df = kline_dict.get(code)
                if df is None:
                    continue
                df_before = df[df[date_col].astype(str).str[:10] <= today]
                if len(df_before) < 60:
                    continue
                info = _safe_single_stock_factor(df_before)
                if info is not None:
                    day_infos[code] = info

            candidates = []
            for code, info in day_infos.items():
                filter_stats["total"] += 1

                # 分级统计
                if info["avg_amount_20"] >= 2_0000_0000 and info["turnover_rate"] >= 1.0:
                    filter_stats["l1_pass"] += 1
                    if info["ma5"] > info["ma10"] > info["ma20"] and info["close"] > info["ma20"]:
                        filter_stats["l2_pass"] += 1
                        if 1.0 <= info["day_change_pct"] <= 9.0 and info["vol_ratio"] >= 1.1:
                            filter_stats["l3_pass"] += 1

                if quick_filter_check(info):
                    prob = 0.5
                    if has_model and info["alpha_factors"]:
                        alpha = info["alpha_factors"]
                        # 只使用模型训练时的特征
                        available = [c for c in feature_cols if c in alpha]
                        if len(available) >= 20:
                            X = np.array([[alpha.get(c, 0.0) for c in available]])
                            try:
                                prob = float(model_info["model"].predict_proba(X)[0, 1])
                                filter_stats["max_prob"] = max(filter_stats["max_prob"], prob)
                            except Exception:
                                prob = 0.5

                    # 综合评分：模型概率主导 + 量价辅助
                    score = prob * 0.7 + (info["day_change_pct"] / 9.0) * 0.15 + (info["vol_ratio"] / 3.0) * 0.15
                    candidates.append((code, score, prob, info["day_change_pct"]))

            # 按评分排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates:
                filter_stats["candidates"] += len(candidates)

            # 概率阈值过滤
            if prob_threshold > 0:
                before = len(candidates)
                candidates = [c for c in candidates if c[2] >= prob_threshold]
                filter_stats["prob_filtered"] += len(candidates)

            # 买入 Top N
            slots = MAX_POSITION_COUNT - len(positions)
            bought = 0
            for code, score, prob, _ in candidates:
                if bought >= slots:
                    break
                if any(p[0] == code for p in positions):
                    continue

                next_day_idx = trade_dates.index(today) + 1 if trade_dates.index(today) < len(trade_dates) - 1 else None
                if next_day_idx is None:
                    continue
                next_day = trade_dates[next_day_idx]

                buy_df = kline_dict.get(code)
                if buy_df is None:
                    continue
                buy_row = buy_df[buy_df[date_col].astype(str).str[:10] == next_day]
                if buy_row.empty:
                    continue
                buy_price = float(buy_row["open"].iloc[0]) * (1 + SLIPPAGE)

                target_amount = initial_capital * SINGLE_STOCK_WEIGHT
                actual_amount = min(target_amount, cash)
                shares = int(actual_amount / buy_price / 100) * 100
                if shares < 100:
                    continue
                cost = shares * buy_price * (1 + COMMISSION_RATE)
                if cost > cash:
                    continue

                cash -= cost
                positions.append((code, next_day, buy_price, shares, cost))
                bought += 1

        # 3. 记录当日净值
        holding_value = cash
        for pos in positions:
            df = kline_dict.get(pos[0])
            if df is not None:
                row = df[df[date_col].astype(str).str[:10] == today]
                if not row.empty:
                    holding_value += pos[3] * float(row["close"].iloc[0])
                else:
                    holding_value += pos[4]
            else:
                holding_value += pos[4]

        daily_values[today] = holding_value

        if (di + 1) % 20 == 0:
            print(f"[回测] 进度: {di+1}/{len(trade_dates)}  "
                  f"净值:{holding_value:,.0f}  持仓:{len(positions)}  交易:{len(closed_trades)}")
            sys.stdout.flush()

    # 清仓
    last_date = trade_dates[-1]
    for pos in positions[:]:
        df = kline_dict.get(pos[0])
        sell_price = pos[2]
        if df is not None:
            row = df[df[date_col].astype(str).str[:10] == last_date]
            if not row.empty:
                sell_price = float(row["close"].iloc[0])
        sell_price_eff = sell_price * (1 - SLIPPAGE)
        revenue = pos[3] * sell_price_eff * (1 - COMMISSION_RATE)
        profit = revenue - pos[4]
        closed_trades.append(profit / pos[4] if pos[4] > 0 else 0)
        cash += revenue
        positions.remove(pos)

    # 打印过滤统计
    total = max(filter_stats["total"], 1)
    print(f"\n[回测] 过滤统计:")
    print(f"  L1(基础)通过:   {filter_stats['l1_pass']:>6} ({filter_stats['l1_pass']/total*100:.1f}%)")
    print(f"  L2(趋势)通过:   {filter_stats['l2_pass']:>6} ({filter_stats['l2_pass']/total*100:.1f}%)")
    print(f"  L3(量价)通过:   {filter_stats['l3_pass']:>6} ({filter_stats['l3_pass']/total*100:.1f}%)")
    print(f"  候选股总数:      {filter_stats['candidates']:>6}")
    print(f"  概率阈值过滤后:  {filter_stats.get('prob_filtered', filter_stats['candidates']):>6}  (阈值={prob_threshold})")
    print(f"  最大模型概率:    {filter_stats['max_prob']:.3f}")

    # ===== 计算指标 =====
    trades_arr = np.array(closed_trades) if closed_trades else np.array([])

    if len(trades_arr) == 0:
        print("\n[回测] 无任何交易")
        return None

    win_rate = float((trades_arr > 0).sum() / len(trades_arr))
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.01
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital

    values_series = pd.Series(daily_values).sort_index()
    peak = values_series.cummax()
    drawdown = (values_series - peak) / peak
    max_drawdown = abs(float(drawdown.min()))

    days = (end_dt - start_dt).days
    years = max(days / 252, 0.01)
    annual_return = (1 + total_return) ** (1 / years) - 1

    daily_ret = values_series.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0

    # ===== 打印报告 =====
    bm = BACKTEST_BENCHMARKS

    print(f"\n{'='*60}")
    print(f"  短线回测报告")
    print(f"  区间: {start_date} ~ {end_date}")
    print(f"{'='*60}")
    print(f"  样本股票池:  {sample_size} 只")
    print(f"  初始资金:    {initial_capital:,.0f}")
    print(f"  最终净值:    {final_value:,.0f}")
    print(f"  总收益率:    {total_return:+.2%}")
    print(f"  年化收益率:  {annual_return:+.2%}  [合格线 >= {bm['annual_return']:.0%}]  {'✓' if annual_return >= bm['annual_return'] else '✗'}")
    print(f"  胜率:        {win_rate:.1%}  [合格线 >= {bm['win_rate']:.0%}]  {'✓' if win_rate >= bm['win_rate'] else '✗'}")
    print(f"  盈亏比:      {profit_loss_ratio:.2f}  [合格线 >= {bm['profit_loss_ratio']:.1f}]  {'✓' if profit_loss_ratio >= bm['profit_loss_ratio'] else '✗'}")
    print(f"  最大回撤:    {max_drawdown:.2%}  [合格线 <= {bm['max_drawdown']:.0%}]  {'✓' if max_drawdown <= bm['max_drawdown'] else '✗'}")
    print(f"  夏普比率:    {sharpe:.2f}  [合格线 >= {bm['sharpe_ratio']:.1f}]  {'✓' if sharpe >= bm['sharpe_ratio'] else '✗'}")
    print(f"  {'─'*56}")
    print(f"  总交易次数:  {len(trades_arr)}")
    print(f"  盈利次数:    {(trades_arr > 0).sum()}")
    print(f"  亏损次数:    {(trades_arr <= 0).sum()}")
    print(f"  平均盈利:    {avg_win:+.2%}")
    print(f"  平均亏损:    {-avg_loss:+.2%}")
    print(f"  最大单笔盈利:{float(trades_arr.max()):+.2%}")
    print(f"  最大单笔亏损:{float(trades_arr.min()):+.2%}")
    print(f"  ─{'─'*55}")
    check_count = sum(1 for k, v in {
        "annual_return": annual_return >= bm["annual_return"],
        "win_rate": win_rate >= bm["win_rate"],
        "profit_loss_ratio": profit_loss_ratio >= bm["profit_loss_ratio"],
        "max_drawdown": max_drawdown <= bm["max_drawdown"],
        "sharpe_ratio": sharpe >= bm["sharpe_ratio"],
    }.items() if v)
    print(f"  达标项: {check_count}/5")
    print(f"{'='*60}\n")

    return {
        "total_return": total_return, "annual_return": annual_return,
        "win_rate": win_rate, "profit_loss_ratio": profit_loss_ratio,
        "max_drawdown": max_drawdown, "sharpe_ratio": sharpe,
        "total_trades": len(trades_arr), "final_value": final_value,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="短线选股回测")
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default="2026-05-01")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--prob", type=float, default=0.0,
                        help="模型概率最低阈值（默认0.0不过滤，建议0.5-0.55）")
    args = parser.parse_args()

    run_backtest(args.start, args.end, args.sample, args.prob)
