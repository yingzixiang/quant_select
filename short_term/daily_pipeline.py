"""
每日选股管线编排
16:00 盘后数据更新 → 17:00 因子计算 → 18:00 选股输出

流程：
1. 获取全A股列表 + 流通市值
2. 批量获取日线数据
3. 计算 Alpha 因子 + 自定义因子
4. 因子预处理（填充/缩尾/标准化）
5. 四级硬约束过滤缩小候选池
6. XGBoost 模型预测上涨概率
7. 多因子综合打分排序
8. 输出 Top 3 选股 + 仓位建议
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from short_term.config import (
    PREDICT_PROB_THRESHOLD, TOP_N_SELECT,
    SINGLE_STOCK_WEIGHT, MAX_POSITION_COUNT,
    FACTOR_SCORE_WEIGHTS,
    SPOT_PRE_FILTER_TOP_N, SPOT_ACTIVITY_WEIGHTS,
)
from base.data_fetch import (
    fetch_a_stock_list, fetch_daily_kline_batch,
    fetch_money_flow_batch, fetch_index_daily, fetch_market_cap_float,
)
from short_term.alpha_factors import compute_alpha_factors_batch
from short_term.custom_factors import compute_all_custom_factors
from short_term.factor_engine import preprocess_factors, compute_factor_composite_score
from short_term.filters import apply_all_filters
from short_term.model_trainer import predict_probability, load_model


def prepare_kline_data(codes: list, target_date: str) -> dict:
    """
    准备日线数据：覆盖训练窗口 + 计算窗口

    Args:
        codes: 股票代码列表
        target_date: 目标日期 YYYYMMDD

    Returns:
        dict: {code: kline_df}
    """
    target = datetime.strptime(target_date, "%Y%m%d")
    today = datetime.now()
    start = target - timedelta(days=120)
    # end 取今天以确保包含最新收盘数据（收盘后数据已落地）
    end = min(target + timedelta(days=10), today)

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    return fetch_daily_kline_batch(codes, start_str, end_str)


def run_daily_selection(target_date: str = None, model_info: dict = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    执行每日选股管线

    Args:
        target_date: 选股日期 YYYYMMDD，默认今天
        model_info: 预加载的模型，None则加载最新
        verbose: 是否打印详细输出

    Returns:
        DataFrame: columns=[code, name, probability, factor_score, final_score, suggestion]
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y%m%d")

    print(f"\n{'='*60}")
    print(f"  短线量化选股管线 - {target_date}")
    print(f"{'='*60}\n")

    # ===== Step 1: 数据获取 =====
    print("[管线] Step 1/6: 获取股票列表和日线数据...")
    sys.stdout.flush()

    stock_list = fetch_a_stock_list()
    if stock_list.empty:
        print("[管线] 错误：无法获取股票列表")
        return pd.DataFrame()

    total_count = len(stock_list)

    # 剔除 ST 股
    st_count = stock_list["name"].str.contains(r"(ST|st)", na=False).sum()
    stock_list = stock_list[~stock_list["name"].str.contains(r"(ST|st)", na=False)]
    total_after_st = len(stock_list)

    # ===== v2.0: Spot 预筛 —— 按活跃度取 Top N 进入深度分析 =====
    has_spot_data = ("amount" in stock_list.columns
                     and stock_list["amount"].notna().sum() > 100)
    if has_spot_data:
        w = SPOT_ACTIVITY_WEIGHTS
        max_amount = stock_list["amount"].max()
        max_turnover = stock_list["turnover_rate"].max()
        max_abs_chg = stock_list["change_pct"].abs().max()

        stock_list["activity_score"] = (
            stock_list["amount"].fillna(0) / max(max_amount, 1) * w["amount"]
            + stock_list["turnover_rate"].fillna(0) / max(max_turnover, 1) * w["turnover_rate"]
            + stock_list["change_pct"].abs().fillna(0) / max(max_abs_chg, 1) * w["abs_change_pct"]
        )
        stock_list = stock_list.sort_values("activity_score", ascending=False)
        top_n = min(SPOT_PRE_FILTER_TOP_N, len(stock_list))
        stock_list = stock_list.head(top_n)
        codes = stock_list["code"].tolist()
        print(f"  📋 预分析: 全A股 {total_count} → 去ST {total_after_st}"
              f" → 活跃度Top{top_n} 候选池 {len(codes)} 只")
    else:
        # v2.1: spot 不可用时的快速预筛 —— 先拉3天日线按成交额排序
        print(f"  📋 预分析: 全A股 {total_count} → 去ST {total_after_st}"
              f" → ⚠ spot数据不可用，使用3天日线快速预筛...")
        sys.stdout.flush()

        all_codes = stock_list["code"].tolist()
        # 快速预筛：腾讯源单线程，慢慢拉，只拉最近3天
        print(f"  [预筛] 腾讯源单线程拉取 Top{min(500,len(all_codes))} 只近3天日线...")
        sys.stdout.flush()
        quick_n = min(500, len(all_codes))
        quick_kline = fetch_daily_kline_batch(
            all_codes[:quick_n],
            (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=5)).strftime("%Y%m%d"),
            target_date,
            max_workers=1       # 单线程，避免触发限流
        )
        # 按近3日平均成交额排序
        ranked = []
        for code, df in quick_kline.items():
            if df is not None and len(df) >= 3:
                if "amount_value" in df.columns:
                    avg_amt = df["amount_value"].astype(float).tail(3).mean()
                else:
                    close = df["close"].astype(float)
                    vol = df["volume"].astype(float)
                    avg_amt = (close.tail(3) * vol.tail(3) * 100).mean()
                ranked.append((code, avg_amt))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if not ranked:
            print("  ⚠ 日线数据源全部不可用（EastMoney + Tencent 均失败），无法选股")
            sys.stdout.flush()
            return pd.DataFrame()
        top_n = min(SPOT_PRE_FILTER_TOP_N, len(ranked))
        codes = [r[0] for r in ranked[:top_n]]
        stock_list = stock_list[stock_list["code"].isin(codes)]
        top_amt = ranked[0][1] / 1e8
        bot_amt = ranked[-1][1] / 1e8
        print(f"  📋 预分析: 3天日线预筛完成 → Top{top_n} 候选池 {len(codes)} 只"
              f"（均成交额 {top_amt:.1f}亿 ~ {bot_amt:.1f}亿）")

    kline_dict = prepare_kline_data(codes, target_date)
    if not kline_dict:
        print("[管线] 错误：无法获取日线数据")
        return pd.DataFrame()

    print(f"  📋 预分析: 日线有效 {len(kline_dict)}/{len(codes)} 只（覆盖率 {len(kline_dict)/len(codes)*100:.1f}%）")
    sys.stdout.flush()

    # 流通市值映射
    cap_map = {}
    for _, row in stock_list.iterrows():
        cap_map[row["code"]] = row.get("float_mv", 0)

    # 资金流向
    money_flow_dict = fetch_money_flow_batch(list(kline_dict.keys()), target_date)
    print(f"  📋 预分析: 资金流向覆盖 {len(money_flow_dict)}/{len(kline_dict)} 只")
    sys.stdout.flush()

    # 指数数据（RPS参考）
    index_df = fetch_index_daily("sh000300",
                                 (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d"),
                                 target_date)
    if not index_df.empty and len(index_df) > 0:
        last_close = index_df["close"].iloc[-1]
        prev_close = index_df["close"].iloc[-2] if len(index_df) > 1 else last_close
        pct = (last_close - prev_close) / prev_close * 100
        print(f"  📋 预分析: 沪深300 最新收盘 {last_close:.1f}（{pct:+.2f}%）")
        sys.stdout.flush()

    # ===== Step 2: 因子计算 =====
    print("\n[管线] Step 2/6: 计算因子...")
    sys.stdout.flush()

    # Alpha因子
    alpha_df = compute_alpha_factors_batch(kline_dict)
    alpha_col_count = len(alpha_df.columns) if not alpha_df.empty else 0
    print(f"  📋 预分析: Alpha 因子 {alpha_col_count} 个，覆盖 {len(alpha_df) if not alpha_df.empty else 0} 只")
    sys.stdout.flush()

    # 自定义因子（量价结构 + 资金流向 + RPS）
    custom_df = compute_all_custom_factors(
        kline_dict=kline_dict,
        money_flow_dict=money_flow_dict,
        index_df=index_df,
    )
    custom_col_count = len(custom_df.columns) if not custom_df.empty else 0
    print(f"  📋 预分析: 自定义因子 {custom_col_count} 个，覆盖 {len(custom_df) if not custom_df.empty else 0} 只")
    sys.stdout.flush()

    # 合并所有因子
    all_factors_list = []
    if not alpha_df.empty:
        all_factors_list.append(alpha_df)
    if not custom_df.empty:
        all_factors_list.append(custom_df)

    if not all_factors_list:
        print("[管线] 错误：无有效因子")
        return pd.DataFrame()

    all_factors = pd.concat(all_factors_list, axis=1, join="outer")
    all_factors = all_factors.loc[:, ~all_factors.columns.duplicated()]

    print(f"  📋 预分析: 合并后因子 {len(all_factors.columns)} 个，覆盖 {len(all_factors)} 只")

    # ===== Step 3: 因子预处理 =====
    print("\n[管线] Step 3/6: 因子预处理...")
    sys.stdout.flush()

    processed_factors = preprocess_factors(all_factors, forward_returns=None)
    if processed_factors.empty:
        print("[管线] 错误：预处理后无有效因子")
        return pd.DataFrame()

    # ===== Step 4: 硬约束过滤 =====
    print("\n[管线] Step 4/6: 五级硬约束过滤...")
    sys.stdout.flush()

    # RPS map from custom factors
    rps_map = {}
    if "rps_20d" in custom_df.columns:
        rps_map = custom_df["rps_20d"].to_dict()
    elif "rps_20d" in all_factors.columns:
        rps_map = all_factors["rps_20d"].to_dict()

    # Money trend
    money_trend = {}
    if "money_trend_3d" in custom_df.columns:
        money_trend = custom_df["money_trend_3d"].to_dict()

    passed_codes = apply_all_filters(
        stock_list_df=stock_list,
        kline_dict=kline_dict,
        money_flow_dict=money_flow_dict,
        cap_map=cap_map,
        rps_map=rps_map,
        money_trend_3d=money_trend,
        index_df=index_df,           # v2.0: Level 5 超额收益用
    )

    if not passed_codes:
        print("[管线] 无股票通过硬约束过滤，可能市场环境不适合短线操作")
        return pd.DataFrame()

    # 过滤后的因子矩阵
    filtered_codes = [c for c in passed_codes if c in processed_factors.index]
    if len(filtered_codes) < 3:
        print(f"[管线] 通过过滤且有因子的股票不足3只: {len(filtered_codes)}")
        filtered_codes = list(passed_codes)[:10]  # fallback

    filtered_factors = processed_factors.loc[filtered_codes]

    # ===== Step 5: 模型预测 =====
    print("\n[管线] Step 5/6: XGBoost 预测...")
    sys.stdout.flush()

    if model_info is None:
        model_info = load_model()

    prob_df = pd.DataFrame()
    if model_info is not None and "model" in model_info:
        feature_cols = model_info.get("feature_cols", [])
        available_features = [c for c in feature_cols if c in filtered_factors.columns]

        if len(available_features) >= 10:
            X_pred = filtered_factors[available_features].values
            codes_pred = filtered_factors.index.tolist()
            prob_df = predict_probability(model_info["model"], X_pred, codes_pred)
            print(f"[管线] 模型预测完成，覆盖 {len(prob_df)} 只")
        else:
            print(f"[管线] 可用特征不足: {len(available_features)}，跳过模型预测")
    else:
        print("[管线] 无可用模型，使用纯因子打分")

    # ===== Step 6: 综合打分排序 =====
    print("\n[管线] Step 6/6: 综合打分排序...")
    sys.stdout.flush()

    # 因子综合得分
    factor_score = compute_factor_composite_score(filtered_factors)

    # v2.0: 趋势强度得分
    trend_score_map = {}
    if "trend_strength_score" in filtered_factors.columns:
        trend_score_map = filtered_factors["trend_strength_score"].to_dict()
        print(f"[管线] 趋势强度得分覆盖 {len(trend_score_map)} 只")

    # 综合打分
    result_rows = []
    for code in filtered_codes:
        row = {"code": code}

        # 股票名称
        name_match = stock_list[stock_list["code"] == code]
        row["name"] = name_match.iloc[0]["name"] if not name_match.empty else ""

        # 模型概率
        if not prob_df.empty:
            prob_match = prob_df[prob_df["code"] == code]
            row["probability"] = float(prob_match["probability"].iloc[0]) if not prob_match.empty else 0.0
        else:
            row["probability"] = 0.0

        # 因子得分
        row["factor_score"] = float(factor_score.get(code, 0))

        # v2.0 趋势强度得分（0-1 归一化）
        trend_s = trend_score_map.get(code, 0.5)
        row["trend_strength"] = trend_s

        # 综合得分 = 模型概率 * 0.55 + 因子得分 * 0.30 + 趋势强度 * 0.15
        factor_score_norm = (row["factor_score"] + 3) / 6  # 粗略归一化到 0~1
        factor_score_norm = max(0, min(1, factor_score_norm))
        row["final_score"] = (row["probability"] * 0.55
                              + factor_score_norm * 0.30
                              + trend_s * 0.15)

        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    if result.empty:
        return result

    result = result.sort_values("final_score", ascending=False)

    # 模型概率筛选
    if model_info is not None and (result["probability"] > 0).any():
        prob_filtered = result[result["probability"] >= PREDICT_PROB_THRESHOLD]

        # v2.3: 模型全部卡死 → 回退到纯因子+趋势评分
        if prob_filtered.empty:
            print(f"  ⚠ 模型概率全低于 {PREDICT_PROB_THRESHOLD}，回退到因子+趋势评分")
            # 按无模型权重重新算：0.55 * factor + 0.45 * trend
            result["final_score"] = (result["factor_score"].apply(
                lambda x: max(0, min(1, (x + 3) / 6))) * 0.55
                + result["trend_strength"].fillna(0.5) * 0.45)
            result = result.sort_values("final_score", ascending=False)
        else:
            result = prob_filtered

    # Top N
    result = result.head(TOP_N_SELECT)

    # 计算交易日期（跳过周末粗略估算）
    from short_term.config import HOLD_DAYS
    target_dt = datetime.strptime(target_date, "%Y%m%d")

    def _next_trade_day(dt, offset_days):
        """粗略估算交易日：跳过周末"""
        d = dt + timedelta(days=offset_days)
        while d.weekday() >= 5:  # 周六=5 周日=6
            d = d + timedelta(days=1)
        return d

    buy_date = _next_trade_day(target_dt, 1)      # T+1 开盘买入
    sell_date = _next_trade_day(buy_date, HOLD_DAYS)  # T+N 收盘卖出
    stop_loss_pct = -5.0                           # 止损线 -5%
    stop_profit_pct = 10.0                         # 止盈线 +10%

    # 丰富结果字段
    result["suggestion"] = [f"买入 {SINGLE_STOCK_WEIGHT*100:.0f}% 仓位" for _ in range(len(result))]
    result["buy_date"] = buy_date.strftime("%Y-%m-%d")
    result["buy_time"] = "09:30 开盘买入"
    result["sell_date"] = sell_date.strftime("%Y-%m-%d")
    result["sell_time"] = "15:00 收盘卖出"
    result["stop_loss"] = f"{stop_loss_pct:+.0f}% 严格止损"
    result["stop_profit"] = f"{stop_profit_pct:+.0f}% 可止盈"
    result["position"] = [f"{SINGLE_STOCK_WEIGHT*100:.0f}%" for _ in range(len(result))]

    # ===== 输出 =====
    print(f"\n{'='*60}")
    print(f"  短线选股结果 — {target_date}")
    print(f"{'='*60}")
    print(f"  选股日期:  {target_date} (今日收盘后)")
    print(f"  买入时间:  {buy_date.strftime('%Y-%m-%d')} 09:30 开盘买入")
    print(f"  卖出时间:  {sell_date.strftime('%Y-%m-%d')} 15:00 收盘卖出")
    print(f"  止损线:    {stop_loss_pct:+.0f}%（盘中跌破立刻卖出）")
    print(f"  止盈线:    {stop_profit_pct:+.0f}%（达到可择机卖出）")
    print(f"  持仓周期:  T+{HOLD_DAYS}")
    print(f"  {'─'*56}")
    if result.empty:
        print(f"  (无符合条件的标的)")
    for i, (_, row) in enumerate(result.iterrows()):
        trend_str = f"    趋势强度: {row['trend_strength']:.3f}" if 'trend_strength' in row else ""
        print(f"  {i+1}. {row['code']} {row['name']}")
        print(f"     上涨概率: {row['probability']:.1%}    因子得分: {row['factor_score']:.3f}{trend_str}")
        print(f"     综合评分: {row['final_score']:.3f}   仓位: {row['position']}")
    print(f"  {'─'*56}")
    print(f"  操作清单:")
    for i, (_, row) in enumerate(result.iterrows()):
        print(f"    {i+1}. {row['buy_date']} 09:30 买入 {row['code']} {row['name']} "
              f"→ {row['sell_date']} 收盘卖出")
        print(f"       止损 {row['stop_loss']} / 止盈 {row['stop_profit']}")
    print(f"{'='*60}\n")

    sys.stdout.flush()
    return result


def save_selection_result(result: pd.DataFrame, target_date: str):
    """保存选股结果"""
    import json
    import os

    output_dir = "short_term/output"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "date": target_date,
        "generated_at": datetime.now().isoformat(),
        "selections": result.to_dict(orient="records"),
    }

    path = os.path.join(output_dir, f"selection_{target_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[管线] 结果已保存至 {path}")
