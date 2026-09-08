"""
月度再平衡编排：月末准备 + 月初执行
"""
import json
import os
from datetime import datetime

import akshare as ak
from config.settings import (
    REBALANCE_MONTH, BENCHMARK,
    STOCK_ALLOC_FLOOR, SELECT_STOCK_NUM,
    HK_MAX_COUNT,
)
from core.macro_timing import (
    fetch_all_macro_indicators, compute_macro_score, derive_allocation,
    get_style_indices_returns, determine_style_allocation, print_macro_report,
)
from core.stock_pool import get_total_stocks, get_hk_stocks, filter_risk_stocks
from core.risk_control import pre_filter_stocks, post_check_portfolio
from core.micro_factors import compute_all_micro_factors
from core.selector import compute_micro_scores, select_target_stocks_v2
from core.factor_calc import get_hk_stock_factors


def run_monthly_rebalance(
    target_month: str = None,
    manual_macro: dict = None,
    financial_date: str = "20260331",
    price_start: str = "20260201",
    price_end: str = "20260526",
):
    """
    执行完整的月度再平衡流程

    Args:
        target_month: 目标月份 e.g. "2026-05"
        manual_macro: 手动覆盖宏观指标 {"monetary_policy": 8, "fiscal_policy": 6, ...}
        financial_date: 财报截止日期
        price_start: 量价因子计算起始日（YYYYMMDD，多期回测时按月度前推）
        price_end: 量价因子计算截止日（YYYYMMDD，多期回测时按月滚动）

    Returns:
        dict: 包含宏观/微观/组合结果的完整状态
    """
    month = target_month or REBALANCE_MONTH
    print(f"\n{'='*60}")
    print(f"  量化选股月度再平衡 — {month}")
    print(f"{'='*60}")

    # ===== Phase 1: 宏观择时 =====
    print("\n[Phase 1/4] 宏观择时...")
    indicator_scores = fetch_all_macro_indicators(manual_macro)
    macro_score = compute_macro_score(indicator_scores)
    stock_weight, bond_weight, contrarian_reasons = derive_allocation(macro_score, indicator_scores, target_month)
    style_returns = get_style_indices_returns()
    style_allocation = determine_style_allocation(style_returns, indicator_scores)
    print_macro_report(indicator_scores, macro_score, stock_weight, bond_weight, style_allocation, contrarian_reasons)

    # ===== Phase 2: 股票池 + 前置风控 =====
    print("\n[Phase 2/5] 构建股票池 + 前置风控...")
    stock_df = get_total_stocks()
    print(f"  A股全市场: {len(stock_df)} 只")
    valid_codes = filter_risk_stocks(stock_df)
    print(f"  ST过滤后: {len(valid_codes)} 只")
    filtered_df, filter_stats = pre_filter_stocks(stock_df)
    print(f"  前置风控: 跳过审计/上市时间检查（无批量API），ST已过滤")

    # 港股池
    hk_stock_df = get_hk_stocks(top_n=200)
    print(f"  港股池(成交额Top 200): {len(hk_stock_df)} 只")

    # ===== Phase 3: A股微观因子 + 打分 =====
    print("\n[Phase 3/5] A股微观多因子计算...")
    factor_df = compute_all_micro_factors(
        valid_codes,
        financial_date=financial_date,
        price_start=price_start,
        price_end=price_end,
    )
    if len(factor_df) == 0:
        print("  [错误] 因子计算无有效数据！")
        return None

    scored_df = compute_micro_scores(factor_df)
    print(f"  A股打分完成，有效: {len(scored_df)} 只，最高分: {scored_df['total_score'].max():.4f}")

    # ===== Phase 4: 港股因子 + 打分 =====
    print("\n[Phase 4/5] 港股多因子计算...")
    hk_factor_df = get_hk_stock_factors(hk_stock_df)
    hk_scored = _score_hk_stocks(hk_factor_df)
    print(f"  港股有效: {len(hk_scored)} 只，最高分: {hk_scored['total_score'].max():.4f}" if len(hk_scored) > 0 else "  港股无有效标的")

    # ===== Phase 5: 选股 + 后置风控 =====
    print("\n[Phase 5/5] 分层选股 + 后置风控...")

    # 港股先选（从港股打分池取前N名）
    hk_selected = []
    hk_max = min(HK_MAX_COUNT, len(hk_scored))
    if hk_max > 0:
        hk_selected = hk_scored.head(hk_max)["code"].tolist()
        print(f"  港股入选: {len(hk_selected)} 只")

    # A股选股（总数扣除港股席位）
    a_target_count = max(SELECT_STOCK_NUM - len(hk_selected), 20)
    a_codes = select_target_stocks_v2(
        scored_df,
        style_allocation,
        total_count=a_target_count,
    )
    target_codes = hk_selected + a_codes
    print(f"  A股入选: {len(a_codes)} 只，合计: {len(target_codes)} 只")

    # 行业映射
    a_industry_map = dict(zip(factor_df["code"], factor_df["industry"].fillna("其他")))
    industry_map = {**a_industry_map}
    for code in hk_selected:
        industry_map[code] = "港股"

    # 后置校验（仅A股部分）
    check_result = post_check_portfolio(a_codes, factor_df, a_industry_map)

    # 输出结果
    print("\n" + "=" * 60)
    print(f"  最终精选组合（{len(target_codes)} 只，A股{len(a_codes)} + 港股{len(hk_selected)}）")
    print("=" * 60)
    a_name_map = dict(zip(stock_df["code"], stock_df["name"]))
    hk_name_map = dict(zip(hk_stock_df["code"], hk_stock_df["name"])) if len(hk_stock_df) > 0 else {}
    name_map = {**a_name_map, **hk_name_map}

    for i, code in enumerate(target_codes, 1):
        name = name_map.get(code, "未知")
        ind = industry_map.get(code, "未知")
        market = "港" if code in hk_selected else "A"
        if market == "港":
            row = hk_scored[hk_scored["code"] == code]
        else:
            row = scored_df[scored_df["code"] == code]
        score = row["total_score"].values[0] if len(row) > 0 and "total_score" in row.columns else 0
        print(f"  {i:2d}. [{market}] {code} {name:<8s} | {ind:<8s} | score={score:.4f}")

    print(f"\n  股票仓位: {stock_weight:.1%} | 债券仓位: {bond_weight:.1%}")
    print(f"  风格: 大盘{style_allocation['large']:.0%} 小盘{style_allocation['small']:.0%} | "
          f"价值{style_allocation['value']:.0%} 成长{style_allocation['growth']:.0%}")

    if check_result["violations"]:
        print(f"\n  [风控违规]:")
        for v in check_result["violations"]:
            print(f"    - {v}")
    if check_result["warnings"]:
        print(f"  [风控提醒]:")
        for w in check_result["warnings"]:
            print(f"    - {w}")

    # 构建portfolio详情
    portfolio_details = []
    for i, code in enumerate(target_codes, 1):
        name = name_map.get(code, "未知")
        ind = industry_map.get(code, "未知")
        market = "港" if code in hk_selected else "A"
        if market == "港":
            row = hk_scored[hk_scored["code"] == code]
        else:
            row = scored_df[scored_df["code"] == code]
        score = row["total_score"].values[0] if len(row) > 0 and "total_score" in row.columns else 0
        pb = row["pb"].values[0] if len(row) > 0 and "pb" in row.columns else None
        roe = row["roe"].values[0] if len(row) > 0 and "roe" in row.columns else None
        portfolio_details.append({
            "rank": i, "code": code, "name": name, "industry": ind,
            "score": score, "pb": pb, "roe": roe, "market": market,
        })

    # 保存JSON快照
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    snapshot = {
        "month": month,
        "macro_score": macro_score,
        "stock_weight": stock_weight,
        "bond_weight": bond_weight,
        "style_allocation": style_allocation,
        "portfolio": target_codes,
        "portfolio_details": portfolio_details,
        "post_check": check_result,
    }
    with open(f"{output_dir}/portfolio_{month}.json", "w") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\n  快照已保存: {output_dir}/portfolio_{month}.json")

    # 保存中文报告
    report_path = save_chinese_report(
        month, indicator_scores, macro_score, stock_weight, bond_weight,
        style_allocation, portfolio_details, check_result,
        output_dir, contrarian_reasons,
    )
    print(f"  中文报告已保存: {report_path}")

    return snapshot


def _compute_prev_month_performance(current_month: str) -> dict:
    """计算上个月精选组合至今的表现"""
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    try:
        cur = datetime.strptime(current_month, "%Y-%m")
        prev = cur - relativedelta(months=1)
        prev_month = prev.strftime("%Y-%m")
    except Exception:
        return None

    # 加载上期精选
    portfolio_path = os.path.join("output", f"portfolio_{prev_month}.json")
    if not os.path.exists(portfolio_path):
        return None

    with open(portfolio_path) as f:
        snap = json.load(f)
    codes = snap.get("portfolio", [])
    if not codes:
        return None

    a_codes = [c for c in codes if len(str(c)) == 6]
    hk_codes = [c for c in codes if len(str(c)) == 5]

    # 获取股票名称映射
    import akshare as ak
    import time
    import numpy as np
    name_map = {}
    try:
        stock_df = ak.stock_info_a_code_name()
        for _, row in stock_df.iterrows():
            name_map[str(row["code"]).zfill(6)] = str(row.get("name", ""))
    except Exception:
        pass

    # ── A股行情 ──
    a_results = []
    for code in a_codes:
        prefix = "sh" if code.startswith("6") else "sz"
        try:
            df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="")
            if df is not None and len(df) >= 25:
                close = df["close"].astype(float)
                start_idx = max(0, len(close) - 40)
                start_p = float(close.iloc[start_idx])
                end_p = float(close.iloc[-1])
                high_p = float(close.iloc[start_idx:].max())
                low_p = float(close.iloc[start_idx:].min())
                chg = round((end_p - start_p) / start_p * 100, 2)
                a_results.append({
                    "code": code, "name": name_map.get(code, ""),
                    "start": round(start_p, 2), "end": round(end_p, 2),
                    "chg_pct": chg, "high": round(high_p, 2), "low": round(low_p, 2),
                })
        except Exception:
            pass
        time.sleep(0.15)

    # ── 港股行情 ──
    hk_results = []
    try:
        hk_spot = ak.stock_hk_spot()
        hk_name = {}
        for _, row in hk_spot.iterrows():
            hk_name[str(row.get("代码", ""))] = str(row.get("中文名称", ""))
    except Exception:
        hk_name = {}

    for code in hk_codes:
        try:
            df = ak.stock_hk_daily(symbol=code, adjust="")
            if df is not None and len(df) >= 25:
                close = df["close"].astype(float)
                start_idx = max(0, len(close) - 40)
                start_p = float(close.iloc[start_idx])
                end_p = float(close.iloc[-1])
                high_p = float(close.iloc[start_idx:].max())
                low_p = float(close.iloc[start_idx:].min())
                chg = round((end_p - start_p) / start_p * 100, 2)
                hk_results.append({
                    "code": code, "name": hk_name.get(code, ""),
                    "start": round(start_p, 2), "end": round(end_p, 2),
                    "chg_pct": chg, "high": round(high_p, 2), "low": round(low_p, 2),
                })
        except Exception:
            pass
        time.sleep(0.15)

    results = a_results + hk_results

    if not results:
        return None

    a_all = [s for s in results if len(s["code"]) == 6]
    hk_all = [s for s in results if len(s["code"]) == 5]

    avg_chg = round(np.mean([r["chg_pct"] for r in results]), 2)
    a_avg = round(np.mean([s["chg_pct"] for s in a_all]), 2) if a_all else 0
    a_up = sum(1 for s in a_all if s["chg_pct"] > 0)
    a_down = len(a_all) - a_up

    hk_avg = round(np.mean([s["chg_pct"] for s in hk_all]), 2) if hk_all else 0
    hk_up = sum(1 for s in hk_all if s["chg_pct"] > 0)
    hk_down = len(hk_all) - hk_up

    return {
        "month": prev_month,
        "total": len(results),
        "avg_chg": avg_chg,
        "win_rate": round((a_up + hk_up) / len(results) * 100, 1),
        "up_count": a_up + hk_up, "down_count": a_down + hk_down,
        "stocks": sorted(results, key=lambda r: r["chg_pct"], reverse=True),
        "best": max(results, key=lambda r: r["chg_pct"]),
        "worst": min(results, key=lambda r: r["chg_pct"]),
        "a_avg": a_avg, "a_up": a_up, "a_down": a_down,
        "hk_avg": hk_avg, "hk_up": hk_up, "hk_down": hk_down,
    }


def _score_hk_stocks(hk_factor_df):
    """港股因子打分：池内 Z-Score 标准化 + 等权求和"""
    import numpy as np
    import pandas as pd

    if hk_factor_df is None or len(hk_factor_df) == 0:
        return pd.DataFrame()

    df = hk_factor_df.copy()
    factor_cols = ["pb", "roe", "profit_growth", "mom_60"]
    directions = {"pb": "negative", "roe": "positive", "profit_growth": "positive", "mom_60": "positive"}

    for col in factor_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        std = series.std(ddof=0)
        std = max(std, 1e-9)
        z = (series - series.mean()) / std
        if directions.get(col) == "negative":
            z = -z
        df[f"{col}_norm"] = z

    norm_cols = [f"{c}_norm" for c in factor_cols if f"{c}_norm" in df.columns]
    if norm_cols:
        df["total_score"] = df[norm_cols].mean(axis=1)
    else:
        df["total_score"] = 0.0

    df["industry"] = "港股"
    return df.sort_values("total_score", ascending=False).reset_index(drop=True)


def save_chinese_report(month, indicator_scores, macro_score, stock_weight, bond_weight,
                        style_allocation, portfolio_details, check_result, output_dir,
                        contrarian_reasons=None):
    """生成中文可读报告（含上期精选绩效追踪）"""
    contrarian_reasons = contrarian_reasons or []

    # 计算上期精选表现
    prev_perf = _compute_prev_month_performance(month)

    lines = []
    lines.append(f"# 量化选股月度再平衡报告")
    lines.append(f"")
    lines.append(f"**调仓月份**: {month}  ")
    lines.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"")

    # ── 一、宏观择时 ──
    lines.append(f"## 一、宏观择时")
    lines.append(f"")
    lines.append(f"### 1.1 宏观综合评分")
    lines.append(f"")
    macro_pct = macro_score * 100
    if macro_pct >= 70:
        macro_level = "偏乐观（利好权益）"
    elif macro_pct >= 50:
        macro_level = "中性偏正面"
    elif macro_pct >= 30:
        macro_level = "中性偏谨慎"
    else:
        macro_level = "偏悲观（防御为主）"
    lines.append(f"- **宏观总分**: {macro_score:.3f}（{macro_pct:.1f} 分）→ 宏观环境 **{macro_level}**")
    lines.append(f"")

    # 分项得分
    groups = {
        "经济增长": ["pmi", "industrial_production"],
        "通胀物价": ["cpi", "ppi", "commodity_index"],
        "流动性与信用": ["bond_10y", "m2", "social_financing", "credit_spread"],
        "外部跨境": ["cn_us_spread", "fx_rate", "northbound_flow"],
        "市场情绪": ["market_volume", "margin_balance"],
        "市场热度": ["turnover_extreme", "advance_decline_ratio", "erp"],
        "风格估值": ["size_pe_percentile", "growth_pe_percentile", "pe_spread"],
        "政策与信心": ["monetary_policy", "fiscal_policy", "meeting_tone", "confidence"],
    }
    indicator_names = {
        "pmi": "制造业PMI", "industrial_production": "工业增加值",
        "cpi": "CPI同比", "ppi": "PPI同比", "commodity_index": "大宗商品价格",
        "bond_10y": "10年期国债", "m2": "M2货币供应", "social_financing": "社融增量",
        "credit_spread": "信用利差",
        "cn_us_spread": "中美利差", "fx_rate": "人民币汇率", "northbound_flow": "北向资金",
        "market_volume": "全市场成交额", "margin_balance": "两融余额",
        "turnover_extreme": "量比极值(热/冷)", "advance_decline_ratio": "涨跌比20日均",
        "erp": "权益风险溢价",
        "size_pe_percentile": "上证PE分位(大盘估值)", "growth_pe_percentile": "创业板PE分位(成长估值)",
        "pe_spread": "成长/价值PE比值",
        "monetary_policy": "货币政策", "fiscal_policy": "财政政策",
        "meeting_tone": "会议定调", "confidence": "消费者信心(代理)",
    }

    for group_name, indicators in groups.items():
        lines.append(f"| 指标 | 原始值 | 得分(0-1) | 状态 |")
        lines.append(f"|------|--------|-----------|------|")
        for name in indicators:
            info = indicator_scores.get(name, {})
            raw = info.get("raw_value")
            score = info.get("score", 0)
            manual = info.get("manual", False)
            proxy = info.get("proxy", False)
            if manual:
                status = "手动输入"
            elif proxy:
                status = "代理计算"
            elif info.get("stale"):
                stale_info = info.get("stale_info", "")
                status = f"自动获取(⚠️过期: {stale_info})" if stale_info else "自动获取(⚠️过期)"
            else:
                status = "自动获取" if info.get("available") else "缺失"
            raw_str = f"{raw:.2f}" if isinstance(raw, (int, float)) else "N/A"
            cn_name = indicator_names.get(name, name)
            lines.append(f"| {cn_name} | {raw_str} | {score:.1f} | {status} |")
        lines.append(f"")

    # ── 1.2 股债配比 ──
    lines.append(f"### 1.2 股债仓位配比")
    lines.append(f"")
    raw_w = STOCK_ALLOC_FLOOR + (1.0 - STOCK_ALLOC_FLOOR) * macro_score
    lines.append(f"- **股票仓位**: {stock_weight:.1%}（基准 {raw_w:.1%}，范围 {STOCK_ALLOC_FLOOR:.0%}~100%）")
    lines.append(f"- **债券仓位**: {bond_weight:.1%}")
    lines.append(f"- **换算逻辑**: 股票仓位 = {STOCK_ALLOC_FLOOR:.0%} + {1-STOCK_ALLOC_FLOOR:.0%} × 宏观总分")
    if contrarian_reasons:
        lines.append(f"- **⚡ 逆向修正**:")
        for r in contrarian_reasons:
            lines.append(f"  - {r}")
    lines.append(f"")

    # ── 1.3 风格轮动 ──
    lines.append(f"### 1.3 市场风格判定")
    lines.append(f"")
    large = style_allocation.get("large", 0.5)
    small = style_allocation.get("small", 0.5)
    value = style_allocation.get("value", 0.5)
    growth = style_allocation.get("growth", 0.5)

    if large > 0.55:
        size_desc = "大盘占优（沪深300近60日跑赢中证1000超2%）"
    elif small > 0.55:
        size_desc = "小盘占优（中证1000近60日跑赢沪深300超2%）"
    else:
        size_desc = "风格均衡（大小盘收益差在±2%以内）"

    if value > 0.55:
        vg_desc = "价值占优（300价值近60日跑赢300成长超2%）"
    elif growth > 0.55:
        vg_desc = "成长占优（300成长近60日跑赢300价值超2%）"
    else:
        vg_desc = "风格均衡（价值成长收益差在±2%以内）"

    lines.append(f"| 维度 | 判定 | 大盘/价值比例 | 小盘/成长比例 |")
    lines.append(f"|------|------|---------------|---------------|")
    lines.append(f"| 大小盘 | {size_desc} | 大盘 {large:.0%} | 小盘 {small:.0%} |")
    lines.append(f"| 价值成长 | {vg_desc} | 价值 {value:.0%} | 成长 {growth:.0%} |")
    lines.append(f"")

    # ── 1.4 上期精选绩效追踪 ──
    if prev_perf:
        import numpy as np
        lines.append(f"### 1.4 上期精选组合绩效追踪（{prev_perf['month']} → 至今）")
        lines.append(f"")

        # 汇总——分市场
        a_stats = prev_perf.get("a_avg", prev_perf["avg_chg"])
        hk_stats = prev_perf.get("hk_avg")

        a_wr = round(prev_perf["a_up"] / (prev_perf["a_up"] + prev_perf["a_down"]) * 100, 1) if (prev_perf["a_up"] + prev_perf["a_down"]) > 0 else 0
        a_icon = "🟢" if a_wr >= 50 else "🔴"
        lines.append(f"- **A股（{prev_perf['a_up']+prev_perf['a_down']}只）**: 平均 {a_stats:+.2f}%　{a_icon} 胜率 {a_wr:.1f}%（{prev_perf['a_up']}涨{prev_perf['a_down']}跌）")

        if hk_stats is not None:
            hk_wr = round(prev_perf["hk_up"] / (prev_perf["hk_up"] + prev_perf["hk_down"]) * 100, 1) if (prev_perf["hk_up"] + prev_perf["hk_down"]) > 0 else 0
            hk_icon = "🟢" if hk_wr >= 50 else "🔴"
            lines.append(f"- **港股（{prev_perf['hk_up']+prev_perf['hk_down']}只）**: 平均 {hk_stats:+.2f}%　{hk_icon} 胜率 {hk_wr:.1f}%（{prev_perf['hk_up']}涨{prev_perf['hk_down']}跌）")

        lines.append(f"- **最佳**: {prev_perf['best']['code']} {prev_perf['best'].get('name','')} {prev_perf['best']['chg_pct']:+.2f}%　　**最差**: {prev_perf['worst']['code']} {prev_perf['worst'].get('name','')} {prev_perf['worst']['chg_pct']:+.2f}%")
        lines.append(f"")

        # 按市场拆分
        a_list = [s for s in prev_perf["stocks"] if len(s["code"]) == 6]
        hk_list = [s for s in prev_perf["stocks"] if len(s["code"]) == 5]

        # A股表
        if a_list:
            lines.append(f"#### A股（{len(a_list)}只）")
            lines.append(f"")
            lines.append(f"| 代码 | 名称 | 6月初价 | 当前价 | 涨跌幅 | 期间最高 | 期间最低 |")
            lines.append(f"|------|------|------|------|------|------|------|")
            for s in sorted(a_list, key=lambda r: r["chg_pct"], reverse=True):
                icon = "🟢" if s["chg_pct"] > 0 else "🔴"
                lines.append(f"| {s['code']} | {s.get('name','-')} | {s['start']} | {s['end']} | {icon} {s['chg_pct']:+.2f}% | {s['high']} | {s['low']} |")
            lines.append(f"")

        # 港股表
        if hk_list:
            lines.append(f"#### 港股（{len(hk_list)}只）")
            lines.append(f"")
            lines.append(f"| 代码 | 名称 | 6月初价 | 当前价 | 涨跌幅 | 期间最高 | 期间最低 |")
            lines.append(f"|------|------|------|------|------|------|------|")
            for s in sorted(hk_list, key=lambda r: r["chg_pct"], reverse=True):
                icon = "🟢" if s["chg_pct"] > 0 else "🔴"
                lines.append(f"| {s['code']} | {s.get('name','-')} | {s['start']} | {s['end']} | {icon} {s['chg_pct']:+.2f}% | {s['high']} | {s['low']} |")
            lines.append(f"")

        lines.append(f"> 选股为买入持有至今的价格变动，不代表实际组合收益（实际有仓位管理和月度调仓）。")
        lines.append(f"")

    # ── 二、精选组合 ──
    lines.append(f"## 二、精选组合（{len(portfolio_details)} 只）")
    lines.append(f"")
    lines.append(f"| 排名 | 代码 | 名称 | 市场 | 申万行业 | 综合得分 | PB | ROE(%) |")
    lines.append(f"|------|------|------|------|----------|----------|-----|--------|")
    for s in portfolio_details:
        pb_str = f"{s['pb']:.2f}" if s['pb'] is not None else "-"
        roe_str = f"{s['roe']:.2f}" if s['roe'] is not None else "-"
        mkt = s.get("market", "A")
        lines.append(f"| {s['rank']:2d} | {s['code']} | {s['name']} | {mkt} | {s['industry']} | {s['score']:.4f} | {pb_str} | {roe_str} |")
    lines.append(f"")

    # ── 三、风控校验 ──
    lines.append(f"## 三、风控校验")
    lines.append(f"")
    if check_result["passed"]:
        lines.append(f"**校验结果: 通过** ✅  ")
    else:
        lines.append(f"**校验结果: 存在违规项** ⚠️  ")
    lines.append(f"")

    if check_result["violations"]:
        lines.append(f"### 违规项")
        for v in check_result["violations"]:
            lines.append(f"- ⚠️ {v}")
        lines.append(f"")

    if check_result["warnings"]:
        lines.append(f"### 提醒项")
        for w in check_result["warnings"]:
            lines.append(f"- ℹ️ {w}")
        lines.append(f"")

    # ── 四、因子说明 ──
    lines.append(f"## 四、因子体系说明")
    lines.append(f"")
    lines.append(f"### 宏观因子（18项，权重合计30%）")
    lines.append(f"18项宏观指标按月更新，覆盖经济增长、通胀物价、流动性信用、外部跨境、市场情绪、政策信心六大维度。")
    lines.append(f"其中货币政策、财政政策、会议定调、信用利差、消费者信心、社融增量 6 项通过 config/manual_macro.py 手动配置。")
    lines.append(f"")
    lines.append(f"### 微观因子（12项，权重合计70%）")
    lines.append(f"")
    lines.append(f"| 类别 | 因子 | 方向 | 数据来源 | 适用 |")
    lines.append(f"|------|------|------|----------|------|")
    lines.append(f"| 估值 | 市净率(PB) | 越低越好 | 百度估值/港股财务 | A+港 |")
    lines.append(f"| 估值 | 市盈率(PE-TTM) | 越低越好 | EPS×4/当前价 代理 | A股 |")
    lines.append(f"| 估值 | 股息率 | 越高越好 | 个股分红历史 | A股 |")
    lines.append(f"| 质量 | 净资产收益率(ROE) | 越高越好 | 业绩快报/港股财务 | A+港 |")
    lines.append(f"| 质量 | 资产负债率 | 越低越好 | 批量API缺失，中性分 | A股 |")
    lines.append(f"| 质量 | 毛利率同比 | 越高越好 | 业绩快报批量 | A股 |")
    lines.append(f"| 成长 | 净利润增速 | 越高越好 | 业绩快报/港股财务 | A+港 |")
    lines.append(f"| 成长 | 营收增速 | 越高越好 | 业绩快报批量 | A股 |")
    lines.append(f"| 成长 | 研发费用占比增速 | 越高越好 | 无API，中性分 | A股 |")
    lines.append(f"| 量价 | 60日动量 | 越高越好 | 腾讯日线/港股日线 | A+港 |")
    lines.append(f"| 量价 | 30日波动率 | 越低越好 | 腾讯日线数据 | A股 |")
    lines.append(f"| 流动性 | 20日均成交额 | 越高越好 | 腾讯日线数据 | A股 |")
    lines.append(f"")
    lines.append(f"**A股打分**: 各因子按申万一级行业内做 Z-Score 标准化（小行业回退到七大板块），统一方向后加权求和。")
    lines.append(f"**港股打分**: 4项因子（PB/ROE/净利润增速/60日动量）池内 Z-Score 标准化后等权求和，最多入选 {HK_MAX_COUNT} 只。")
    lines.append(f"")

    # Write file
    report_path = os.path.join(output_dir, f"report_{month}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path
